"""Does running speed shift the preferred SF / TF of grating responses? Frame-level model comparison.

For the reviewer grating analysis (`figures/fig_reviewer_gratings.ipynb`). Fits four versions of the
SF/TF/direction model of `fit_gaussian_blob.grating_tuning` to single imaging frames of the SFTF
recordings, with the preferred log SF and log TF allowed to depend linearly on log running speed:

    log SF0 = sf0 + a_sf * r,   log TF0 = tf0 + a_tf * r,   r = log(RS) - median log(RS) of the session

    base: a_sf = a_tf = 0;  gain: g;  gain_tf: g, a_tf;  gain_sf: g, a_sf

The gain multiplies the amplitude by RS^g (log amplitude + g * r). The running-speed tuning models replace
it with a Gaussian in log RS, parameterised and bounded as the RS/OF fits of the pipeline
(`fit_gaussian_blob.fit_rs_of_tuning`: preferred RS 0.005-5 m/s, sigma^2 = exp(log_sigma_rs2) + MIN_SIGMA,
log_sigma_rs2 free):

    log amplitude - (r - r0)^2 / (2 sigma^2)      rsg: r0, log_sigma_rs2;  rsg_tf: + a_tf;  rsg_sf: + a_sf
                                                  rsg_notf: rsg without TF (1D SF + dir + RS);
                                                  rsg_nosf: rsg without SF (1D TF + dir + RS)

with r0 the preferred log RS relative to the session median.

Samples are imaging frames during gratings with RS > RS_THR (used only when log-transforming to cut zeros, not as a definition of static), neural trace shifted by
LAG frames relative to the stimulus and RS (default: 2 frames for deconvolved spikes, 0 for dF/F as the RS/OF fits).
The first SKIP frames of each grating are dropped, as the trial window of the trial-level analyses
(`[onset + 8, next_onset)`, `gratings.generate_trials_df`): they carry the decay of the previous grating's response.
Each model is fitted on all frames (parameters, `rsq`) and cross-validated leaving one recording out
(held-out R2 and Spearman of the concatenated predictions). The running-speed models start from the base
fit with zero slopes, plus random starts.

With --still (prepare), the frames are instead the still frames, as the not-running depth fits of figure 1:
RS < STILL_THR in the frame, the STILL_FRAMES - 1 frames before it (~1 s) and the LAG frames after it (up to
the frame of the signal), wherever these frames fall (previous grating or outside the gratings). They are saved
to OUT_DIR/frames_still/<session>.npz, beside the running frames, and are not fitted (`fit` and `merge` read
only the running frames); the figure uses them for the static curves (`sftf.build_speed_curves`). The RS column
of X is 0. Use --all-cells for both, so that the ROIs match.

Only `iscell` ROIs whose trial-level fit on the same trace is cross-validated SF/TF tuned (one-sided
Spearman p < 0.05) are fitted: production `neurons_df_sftf` for dF/F, `neurons_df_sftf_spks` for
spikes. Production files are only read. Three steps, so that ROIs can be fitted as separate jobs:

    prepare SESSION      sync once, save running frames and traces to OUT_DIR/frames/<session>.npz
                         (SF/TF-tuned iscell ROIs, or all iscell ROIs with --all-cells); with --still,
                         still frames to OUT_DIR/frames_still/<session>.npz
    fit SESSION ROI      fit one ROI -> OUT_DIR/rois/<session>_<roi>.pickle (skipped if it exists;
                         with --models, only the models missing from it are fitted and added)
    merge SESSION        -> OUT_DIR/<session>_sftf_running_models.pickle (one row per ROI) + .json

Usage:
    python sftf_running_models.py {prepare,fit,merge} SESSION [ROI] --out-dir DIR [--n-jobs 1]
        [--trace {spks,dff}] [--lag LAG] [--skip SKIP] [--all-cells] [--still] [--models MODEL ...] [--overwrite]
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from cottage_analysis.analysis import common_utils
from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.preprocessing import synchronisation

PROJECT = "hey2_3d-vision_foodres_20220101"
PHOTODIODE_PROTOCOL = 5
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}
PARAMS = ["SpatialFrequency", "TemporalFrequency", "Angle"]
RS_THR = 0.01  # m/s, floor used only when log-transforming RS to cut 0s (as rs_thr of the RS/OF fits; not a definition of static)
STILL_THR = 0.05  # m/s, max RS of still frames (--still), as figure 1
STILL_FRAMES = 15  # still frame t: RS < STILL_THR from t - 14 (~1 s) to t + lag
LAG = 0  # default frames between stimulus / RS and signal for dF/F (as the RS/OF fits; 2 for deconvolved spikes)
SKIP = 8  # frames dropped after each grating onset (0.53 s at 15.2 Hz, as the trial window of the trial-level fits)
MIN_SIGMA = 0.25
N_ITER_BASE = 5
N_ITER_RS = 3  # random starts of the running-speed models, on top of the base-fit start
SLOPE_BOUND = 5  # |a_sf|, |a_tf| in log SF (TF) per log RS
MODELS = {
    "base": (),
    "gain": ("g",),
    "gain_tf": ("g", "a_tf"),
    "gain_sf": ("g", "a_sf"),
    "rsg": ("r0", "log_sigma_rs2"),
    "rsg_tf": ("r0", "log_sigma_rs2", "a_tf"),
    "rsg_sf": ("r0", "log_sigma_rs2", "a_sf"),
    "rsg_notf": ("r0", "log_sigma_rs2"),
    "rsg_nosf": ("r0", "log_sigma_rs2"),
}
# GratingParams fields omitted in 1D SF-only / TF-only models
DROPPED_GRATING_PARAMS = {
    "rsg_notf": ("tf0", "log_sigma_y2", "theta"),
    "rsg_nosf": ("sf0", "log_sigma_x2", "theta"),
}
# models also started from the fits of the simpler models they contain (shared terms copied, new at 0)
NESTED_STARTS = {
    "gain_tf": ("gain",),
    "gain_sf": ("gain",),
    "rsg": ("gain",),
    "rsg_tf": ("rsg", "gain_tf"),
    "rsg_sf": ("rsg", "gain_sf"),
    "rsg_notf": ("rsg", "gain"),
    "rsg_nosf": ("rsg", "gain"),
}
RS_RANGE = (
    0.005,
    5,
)  # m/s, bounds of the preferred running speed (as rs_min, rs_max of the RS/OF fits)
RSG_START_SIGMA = 5.0  # sigma of the running-speed Gaussian started from a gain fit
RSG_START_QUANTILES = (
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
)  # r0 starts (sigma = 1) at these quantiles of r


def session_frames(flm, session, trace="spks", lag=2, still=False, skip=SKIP):
    """Running (or still) frames during gratings of all SFTF recordings of a session.

    Args:
        flm: Flexilims session.
        session: session name.
        trace: "spks" (deconvolved spikes) or "dff" (dF/F).
        lag: number of frames to shift signal relative to stimulus and RS.
        still: keep the still frames instead of the running frames: RS < STILL_THR in the frame,
            the STILL_FRAMES - 1 before it and the `lag` after it (up to the signal frame), over
            all frames of the recording.
        skip: drop the frames whose stimulus frame is fewer than `skip` frames after the onset of its grating
            (onset = change of the parameter-log time, as the trial tables).

    Returns:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - median log RS (0 if `still`);
        traces (n_frames, n_rois), recording (n_frames,), median log RS (NaN if `still`).
    """
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()
    xs, traces_list, recs = [], [], []
    for _, rec in recordings.iterrows():
        vs_df = synchronisation.generate_vs_df(
            recording=rec,
            photodiode_protocol=PHOTODIODE_PROTOCOL,
            flexilims_session=flm,
            project=PROJECT,
        )
        img = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=rec,
            flexilims_session=flm,
            filter_datasets=FILTER_TRACES,
            return_volumes=True,
            add_spikes=(trace == "spks"),
        ).reset_index(drop=True)
        rs = (img.mouse_z_harp.diff() / img.mouse_z_harptime.diff()).to_numpy()
        stim = img[PARAMS].to_numpy(float)
        tr = np.vstack(img.spks.values if trace == "spks" else img.dffs.values)
        # frames since the onset of the current grating
        harptime = img["stimulus_harptime"]
        onset = (harptime.notna() & (harptime != harptime.shift())).to_numpy()
        frame = np.arange(len(img))
        since_onset = frame - np.maximum.accumulate(np.where(onset, frame, 0))
        is_still = np.zeros(len(rs), bool)
        is_still[
            common_utils.find_thresh_sequence(
                array=rs,
                threshold_max=STILL_THR,
                length=STILL_FRAMES + lag,  # frames t - STILL_FRAMES + 1 to t + lag
                shift=STILL_FRAMES,
            )
        ] = True
        # stimulus and RS of frame t with signal of frame t + lag
        n = len(stim) - lag  # not stim[:-lag], empty for lag 0
        stim, rs, is_still, tr = stim[:n], rs[:n], is_still[:n], tr[lag:]
        after_onset = since_onset[:n] >= skip
        if still:
            keep = np.isfinite(stim).all(axis=1) & after_onset & is_still
            log_rs = np.zeros(keep.sum())
        else:
            keep = (
                np.isfinite(stim).all(axis=1)
                & after_onset
                & np.isfinite(rs)
                & (rs > RS_THR)
            )
            log_rs = np.log(rs[keep])
        xs.append(np.c_[np.log(stim[keep, :2]), np.deg2rad(stim[keep, 2]), log_rs])
        traces_list.append(tr[keep])
        recs.append(np.full(keep.sum(), rec["name"]))
    X = np.concatenate(xs).T
    log_rs_median = np.nan if still else np.median(X[3])
    if not still:
        X[3] -= log_rs_median
    return X, np.concatenate(traces_list), np.concatenate(recs), log_rs_median


def rs_tuning(X, *params, free=(), drop=(), min_sigma=MIN_SIGMA):
    """grating_tuning with log SF0 / log TF0 shifted or amplitude scaled by running speed.

    `drop` lists `GratingParams` fields omitted from `params` (e.g. `("tf0", "log_sigma_y2", "theta")`
    for a 1D SF-only model, or `("sf0", "log_sigma_x2", "theta")` for a 1D TF-only model).
    """
    if drop:
        kept = [f for f in GratingParams._fields if f not in drop]
        g_dict = dict(zip(kept, params[: len(kept)]))
        p = GratingParams(**{f: g_dict.get(f, 0.0) for f in GratingParams._fields})
        slopes = dict(zip(free, params[len(kept) :]))
    else:
        p = GratingParams(*params[:10])
        slopes = dict(zip(free, params[10:]))
    sf0 = p.sf0 + slopes.get("a_sf", 0) * X[3]
    tf0 = p.tf0 + slopes.get("a_tf", 0) * X[3]
    log_amp = p.log_amplitude + slopes.get("g", 0) * X[3]
    if "r0" in slopes:
        sigma2 = np.exp(slopes["log_sigma_rs2"]) + min_sigma
        log_amp = log_amp - (X[3] - slopes["r0"]) ** 2 / (2 * sigma2)
    sf = 0.0 if "sf0" in drop else X[0]
    tf = 0.0 if "tf0" in drop else X[1]
    return grating_tuning(
        (sf, tf, X[2]), log_amp, sf0, tf0, *p[3:], min_sigma=min_sigma
    )


def bounds(X, free, log_rs_median=None, drop=()):
    """Bounds of the parameters; the preferred running speed r0 needs the session `log_rs_median`."""
    lower = GratingParams(
        -np.inf,
        X[0].min() - 1,
        X[1].min() - 1,
        -20.0,
        -20.0,
        0,
        -np.inf,
        0,
        -np.inf,
        0,
    )
    upper = GratingParams(
        np.inf,
        X[0].max() + 1,
        X[1].max() + 1,
        np.inf,
        np.inf,
        np.pi / 2,
        np.inf,
        2 * np.pi,
        6.0,
        1,
    )
    lo_g = [b for f, b in zip(GratingParams._fields, lower) if f not in drop]
    hi_g = [b for f, b in zip(GratingParams._fields, upper) if f not in drop]
    extra = {"log_sigma_rs2": (-np.inf, np.inf)}
    if "r0" in free:
        extra["r0"] = tuple(np.log(RS_RANGE) - log_rs_median)
    lo_hi = [extra.get(t, (-SLOPE_BOUND, SLOPE_BOUND)) for t in free]
    return lo_g + [b[0] for b in lo_hi], hi_g + [b[1] for b in lo_hi]


def project_grating(popt10, drop=()):
    """Project 10-element `GratingParams` onto kept fields when `drop` omits SF or TF."""
    if not drop:
        return list(popt10)
    p = GratingParams(*popt10)
    sx = np.exp(np.clip(p.log_sigma_x2, -20, 20)) + MIN_SIGMA
    sy = np.exp(np.clip(p.log_sigma_y2, -20, 20)) + MIN_SIGMA
    c2, s2 = np.cos(p.theta) ** 2, np.sin(p.theta) ** 2
    s_sf2 = 1.0 / (c2 / sx + s2 / sy)
    s_tf2 = 1.0 / (s2 / sx + c2 / sy)
    p = p._replace(
        log_sigma_x2=float(np.log(max(s_sf2 - MIN_SIGMA, 1e-9))),
        log_sigma_y2=float(np.log(max(s_tf2 - MIN_SIGMA, 1e-9))),
    )
    return [v for f, v in zip(GratingParams._fields, p) if f not in drop]


def nested_start(simpler, popt, free, lo, hi, drop=()):
    """Start of a model with terms `free` (bounds lo, hi) from the fit `popt` of model `simpler`.

    Shared terms are copied, new terms start at 0, and a gain g is converted to a running-speed
    Gaussian of sigma RSG_START_SIGMA with the same slope at r = 0 (r0 = g sigma^2, log amplitude
    raised by r0^2 / 2 sigma^2), r0 clipped to its bounds.
    """
    n_simpler_g = 10 - len(DROPPED_GRATING_PARAMS.get(simpler, ()))
    p = (
        project_grating(popt[:10], drop)
        if n_simpler_g == 10
        else list(popt[:n_simpler_g])
    )
    terms = dict(zip(MODELS[simpler], popt[n_simpler_g:]))
    if "r0" in free and "g" in terms:
        i = len(p) + free.index("r0")
        r0 = float(np.clip(terms.pop("g") * RSG_START_SIGMA**2, lo[i], hi[i]))
        p[0] += r0**2 / (2 * RSG_START_SIGMA**2)
        terms.update(r0=r0, log_sigma_rs2=np.log(RSG_START_SIGMA**2 - MIN_SIGMA))
    return p + [terms.get(t, 0.0) for t in free]


def random_p0(X, y):
    """Random start of the base model, SF0 / TF0 / direction at the best stimulus value."""
    y = pd.Series(y)
    return list(
        GratingParams(
            log_amplitude=np.random.normal(),
            sf0=y.groupby(X[0]).mean().idxmax(),
            tf0=y.groupby(X[1]).mean().idxmax(),
            log_sigma_x2=np.random.normal(),
            log_sigma_y2=np.random.normal(),
            theta=np.random.uniform(0, 0.5 * np.pi),
            offset=np.random.normal(),
            alpha0=y.groupby(X[2]).mean().idxmax(),
            log_kappa=np.random.normal(),
            dsi=np.random.uniform(0, 1),
        )
    )


def best_fit(func, X, y, lo, hi, starts):
    """Best (highest R2) of curve_fit runs from each start; (popt, rsq), NaN if all fail."""
    best = (np.full(len(lo), np.nan), -np.inf)
    for p0 in starts:
        p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)
        try:
            popt, _ = curve_fit(func, X, y, p0=p0, bounds=(lo, hi), maxfev=5000)
        except (RuntimeError, ValueError):
            continue
        rsq = common_utils.calculate_r_squared(y, func(X, *popt))
        if rsq > best[1]:
            best = (popt, rsq)
    return best


def fit_models(X, y, models=None, log_rs_median=None):
    """Models on frames X, y: {model: (popt, rsq)}, running-speed models warm-started.

    `log_rs_median` (log m/s) is needed for the bounds of the running-speed tuning models.

    `models` (default all) restricts which models are fitted; the models they are warm-started from
    are fitted too. Random starts are drawn for every model in MODELS order whether it is fitted or
    not, so a model's fit does not depend on which others are requested.
    """
    models = list(MODELS) if models is None else list(models)
    needed = set(models) | {"base"}
    stack = list(models)
    while stack:  # models needed for warm starts, recursively
        for simpler in NESTED_STARTS.get(stack.pop(), ()):
            if simpler not in needed:
                needed.add(simpler)
                stack.append(simpler)
    np.random.seed(42)
    out = {}
    lo, hi = bounds(X, ())
    out["base"] = best_fit(
        partial(rs_tuning, free=()),
        X,
        y,
        lo,
        hi,
        [random_p0(X, y) for _ in range(N_ITER_BASE)],
    )
    base = out["base"][0] if np.all(np.isfinite(out["base"][0])) else random_p0(X, y)
    for model, free in MODELS.items():
        if model == "base":
            continue
        drop = DROPPED_GRATING_PARAMS.get(model, ())
        random_starts = [
            [v for f, v in zip(GratingParams._fields, random_p0(X, y)) if f not in drop]
            + list(np.random.uniform(-1, 1, len(free)))
            for _ in range(N_ITER_RS)
        ]
        if model not in needed:
            continue
        lo, hi = bounds(X, free, log_rs_median, drop=drop)
        base_proj = project_grating(base, drop)
        starts = [base_proj + [0.0] * len(free)]
        if (
            "r0" in free
        ):  # base fit with a running-speed Gaussian (sigma = 1) at quantiles of r
            starts = [
                base_proj
                + [
                    {
                        "r0": np.quantile(X[3], q),
                        "log_sigma_rs2": np.log(1 - MIN_SIGMA),
                    }.get(t, 0.0)
                    for t in free
                ]
                for q in RSG_START_QUANTILES
            ]
        for simpler in NESTED_STARTS.get(model, ()):
            popt = out[simpler][0]
            if np.all(np.isfinite(popt)):
                starts.append(nested_start(simpler, popt, free, lo, hi, drop=drop))
        out[model] = best_fit(
            partial(rs_tuning, free=free, drop=drop),
            X,
            y,
            lo,
            hi,
            starts + random_starts,
        )
    return {m: out[m] for m in models}


def fit_roi(X, y, recording, n_jobs=1, models=None, log_rs_median=None):
    """Full-data and leave-one-recording-out fits of the models (default all) for one ROI.

    The full-data fit and the folds are independent and run in parallel with `n_jobs`.
    """
    recs = np.unique(recording)
    subsets = [np.ones(len(y), bool)] + [recording != rec for rec in recs]
    if n_jobs == 1:
        fits = [fit_models(X[:, s], y[s], models, log_rs_median) for s in subsets]
    else:
        from joblib import Parallel, delayed

        fits = Parallel(n_jobs=n_jobs)(
            delayed(fit_models)(X[:, s], y[s], models, log_rs_median) for s in subsets
        )
    full, folds = fits[0], fits[1:]
    preds = {m: np.full(len(y), np.nan) for m in full}
    for rec, fold in zip(recs, folds):
        test = recording == rec
        for m in full:
            if np.all(np.isfinite(fold[m][0])):
                preds[m][test] = rs_tuning(
                    X[:, test],
                    *fold[m][0],
                    free=MODELS[m],
                    drop=DROPPED_GRATING_PARAMS.get(m, ()),
                )
    row = {}
    for m in full:
        free = MODELS[m]
        drop = DROPPED_GRATING_PARAMS.get(m, ())
        names = [f for f in GratingParams._fields if f not in drop] + list(free)
        row.update({f"{m}_{n}": v for n, v in zip(names, full[m][0])})
        row[f"{m}_rsq"] = full[m][1]
        ok = np.isfinite(preds[m])
        row[f"{m}_test_rsq"] = common_utils.calculate_r_squared(y[ok], preds[m][ok])
        row[f"{m}_test_spearmanr_rval"], row[f"{m}_test_spearmanr_pval"] = spearmanr(
            y[ok], preds[m][ok]
        )
    return row


def iscell_rois(flm, session):
    """Boolean iscell mask of the suite2p ROIs of a session."""
    rois_ds = flz.get_datasets(
        origin_name=session,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        flexilims_session=flm,
        allow_multiple=False,
    )
    return np.load(rois_ds.path_full / "plane0" / "iscell.npy")[:, 0] == 1


def sftf_tuned_rois(flm, session, trace="spks"):
    """iscell ROIs that are cross-validated SF/TF tuned in the trial-level fit of `trace`.

    The fit is the production neurons_df_sftf.pickle for dF/F, neurons_df_sftf_spks.pickle for
    deconvolved spikes (`rerun_depth_sftf.py --use-col spks`).
    """
    folder = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=flm, project=PROJECT, conflicts="skip"
    ).path_full.parent
    fname = (
        "neurons_df_sftf.pickle"
        if trace == "dff"
        else f"neurons_df_sftf_{trace}.pickle"
    )
    fit = pd.read_pickle(folder / fname)
    iscell = iscell_rois(flm, session)
    pval = common_utils.one_sided_pval_from_spearman(
        fit.sftf_test_spearmanr_rval, fit.sftf_test_spearmanr_pval
    )
    return np.flatnonzero(iscell & (np.asarray(pval) < 0.05))


def frames_path(out_dir, session, still=False):
    return Path(out_dir) / ("frames_still" if still else "frames") / f"{session}.npz"


def prepare(
    session, out_dir, trace="spks", lag=2, all_cells=False, still=False, skip=SKIP
):
    """Sync the session once and save the running frames and the ROIs' signal.

    ROIs: iscell ROIs that are SF/TF tuned in the trial-level fit of `trace`, or all iscell ROIs
    with `all_cells`. Still frames instead of running frames with `still`.
    """
    flm = flz.get_flexilims_session(PROJECT)
    rois = (
        np.flatnonzero(iscell_rois(flm, session))
        if all_cells
        else sftf_tuned_rois(flm, session, trace=trace)
    )
    X, traces, recording, log_rs_median = session_frames(
        flm, session, trace=trace, lag=lag, still=still, skip=skip
    )
    path = frames_path(out_dir, session, still=still)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=X,
        traces=traces[:, rois],
        dff=traces[:, rois],
        rois=rois,
        recording=recording,
        log_rs_median=log_rs_median,
        trace=trace,
        lag=lag,
        still=still,
        skip=skip,
    )
    print(
        f"{session} ({trace}, lag={lag}, skip={skip}): {X.shape[1]} {'still' if still else 'running'} frames, {len(rois)} ROIs -> {path}"
    )


def fit(session, roi, out_dir, n_jobs, models=None, overwrite=False):
    """Fit one ROI from the prepared frames.

    Without `models`, skipped if its result exists. With `models`, those missing from an existing
    result (all of them with `overwrite`) are fitted and their columns added to it, replacing the
    previous columns of these models (the other columns are kept). Running frames only.
    """
    d = np.load(frames_path(out_dir, session))
    assert not ("still" in d and bool(d["still"])), "still frames: not fitted"
    target = Path(out_dir) / "rois" / f"{session}_{roi}.pickle"
    old = pd.read_pickle(target) if target.exists() else None
    if models is None:
        if old is not None:
            print(f"{target} exists, skipping")
            return
    elif old is not None and not overwrite:
        models = [m for m in models if f"{m}_test_rsq" not in old.columns]
        if not models:
            print(f"{target} has all requested models, skipping")
            return
    (col,) = np.flatnonzero(d["rois"] == roi)
    traces = d["traces"] if "traces" in d else d["dff"]
    start = time.time()
    row = fit_roi(
        d["X"],
        traces[:, col],
        d["recording"],
        n_jobs=n_jobs,
        models=models,
        log_rs_median=float(d["log_rs_median"]),
    )
    if old is None:
        out = pd.DataFrame([dict(session=session, roi=roi, **row)])
    else:  # drop previous columns of the refitted models (not those of longer model names)
        longer = [
            o for o in MODELS if any(o != m and o.startswith(f"{m}_") for m in models)
        ]
        stale = [
            c
            for c in old
            if any(c.startswith(f"{m}_") for m in models)
            and not any(c.startswith(f"{o}_") for o in longer if o not in models)
        ]
        out = old.drop(columns=stale).assign(**row)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_pickle(target)
    print(f"{session} ROI {roi} fitted in {time.time() - start:.0f} s -> {target}")


def merge(session, out_dir):
    """Concatenate the ROI results of a session; fails if any ROI is missing."""
    out_dir = Path(out_dir)
    d = np.load(frames_path(out_dir, session))
    paths = [out_dir / "rois" / f"{session}_{roi}.pickle" for roi in d["rois"]]
    missing = [p.name for p in paths if not p.exists()]
    assert not missing, f"{len(missing)} ROIs missing, e.g. {missing[:5]}"
    out = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    out.to_pickle(out_dir / f"{session}_sftf_running_models.pickle")
    lag = int(d["lag"]) if "lag" in d else LAG
    trace = str(d["trace"]) if "trace" in d else "dff"
    skip = (
        int(d["skip"]) if "skip" in d else 0
    )  # frames prepared before the onset skip have none
    (out_dir / f"{session}_sftf_running_models.json").write_text(
        json.dumps(
            dict(
                n_frames=int(d["X"].shape[1]),
                log_rs_median=float(d["log_rs_median"]),
                rs_thr=RS_THR,
                trace=trace,
                lag=lag,
                skip=skip,
                still=bool(d["still"]) if "still" in d else False,
                still_thr=STILL_THR,
                still_frames=STILL_FRAMES,
                filter_traces=FILTER_TRACES,
                n_iter_base=N_ITER_BASE,
                n_iter_rs=N_ITER_RS,
                slope_bound=SLOPE_BOUND,
                models={m: MODELS[m] for m in MODELS if f"{m}_test_rsq" in out},
            ),
            indent=1,
        )
    )
    print(
        f"Saved {len(out)} ROIs to {out_dir / f'{session}_sftf_running_models.pickle'}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["prepare", "fit", "merge"])
    p.add_argument("session")
    p.add_argument("roi", nargs="?", type=int, help="ROI to fit (fit only)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument(
        "--trace",
        choices=["spks", "dff"],
        default="spks",
        help="trace type to extract (prepare only, default: spks)",
    )
    p.add_argument(
        "--lag",
        type=int,
        default=None,
        help="stimulus to trace lag in frames (prepare only, default: 2 for spks, 0 for dff)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=SKIP,
        help=f"frames dropped after each grating onset (prepare only, default: {SKIP})",
    )
    p.add_argument(
        "--all-cells",
        action="store_true",
        help="all iscell ROIs, not only the SF/TF-tuned ones (prepare only)",
    )
    p.add_argument(
        "--still",
        action="store_true",
        help="still frames instead of running frames, saved to frames_still/ and not fitted (prepare only)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        help="fit only these models, adding them to an existing ROI result (fit only)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="refit the --models even if already in the ROI result (fit only)",
    )
    a = p.parse_args()
    if a.command == "prepare":
        lag = a.lag if a.lag is not None else (2 if a.trace == "spks" else LAG)
        prepare(
            a.session,
            a.out_dir,
            trace=a.trace,
            lag=lag,
            all_cells=a.all_cells,
            still=a.still,
            skip=a.skip,
        )
    elif a.command == "fit":
        fit(a.session, a.roi, a.out_dir, a.n_jobs, a.models, a.overwrite)
    else:
        merge(a.session, a.out_dir)
