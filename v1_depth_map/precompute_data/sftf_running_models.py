"""Does running speed shift the preferred SF / TF of grating responses? Frame-level model comparison.

For the reviewer grating analysis (`figures/fig_reviewer_gratings.ipynb`). Fits four versions of the
SF/TF/direction model of `fit_gaussian_blob.grating_tuning` to single imaging frames of the SFTF
recordings, with the preferred log SF and log TF allowed to depend linearly on log running speed:

    log SF0 = sf0 + a_sf * r,   log TF0 = tf0 + a_tf * r,   r = log(RS) - median log(RS) of the session

    base: a_sf = a_tf = 0;  gain: g;  gain_tf: g, a_tf;  gain_sf: g, a_sf

Samples are imaging frames during gratings with RS > RS_THR (static frames excluded), neural trace shifted by
LAG frames relative to the stimulus and RS (default: 2 frames for deconvolved spikes, 5 frames for dF/F).
Each model is fitted on all frames (parameters, `rsq`) and cross-validated leaving one recording out
(held-out R2 and Spearman of the concatenated predictions). The running-speed models start from the base
fit with zero slopes, plus random starts.

Only `iscell` ROIs whose production `neurons_df_sftf` fit is cross-validated SF/TF tuned (one-sided
Spearman p < 0.05) are fitted. Production files are only read. Three steps, so that ROIs can be fitted
as separate jobs:

    prepare SESSION      sync once, save running frames and traces to OUT_DIR/frames/<session>.npz
    fit SESSION ROI      fit one ROI -> OUT_DIR/rois/<session>_<roi>.pickle (skipped if it exists;
                         with --models, only the models missing from it are fitted and added)
    merge SESSION        -> OUT_DIR/<session>_sftf_running_models.pickle (one row per ROI) + .json

Usage:
    python sftf_running_models.py {prepare,fit,merge} SESSION [ROI] --out-dir DIR [--n-jobs 1]
        [--trace {spks,dff}] [--lag LAG] [--models MODEL ...]
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
RS_THR = 0.01  # m/s, frames at or below are static (as rs_thr of the RS/OF fits)
LAG = 5  # default frames between stimulus / RS and signal (5 for dF/F, 2 for deconvolved spikes)
MIN_SIGMA = 0.25
N_ITER_BASE = 5
N_ITER_RS = 3  # random starts of the running-speed models, on top of the base-fit start
SLOPE_BOUND = 5  # |a_sf|, |a_tf| in log SF (TF) per log RS
MODELS = {
    "base": (),
    "gain": ("g",),
    "gain_tf": ("g", "a_tf"),
    "gain_sf": ("g", "a_sf"),
}
# models also started from the fits of the simpler models they contain (new terms at 0)
NESTED_STARTS = {"gain_tf": ("gain",), "gain_sf": ("gain",)}


def session_frames(flm, session, trace="spks", lag=2):
    """Running frames during gratings of all SFTF recordings of a session.

    Args:
        flm: Flexilims session.
        session: session name.
        trace: "spks" (deconvolved spikes) or "dff" (dF/F).
        lag: number of frames to shift signal relative to stimulus and RS.

    Returns:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - median log RS;
        traces (n_frames, n_rois), recording (n_frames,), median log RS.
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
        # stimulus and RS of frame t with signal of frame t + lag
        stim, rs, tr = stim[:-lag], rs[:-lag], tr[lag:]
        keep = np.isfinite(stim).all(axis=1) & np.isfinite(rs) & (rs > RS_THR)
        xs.append(
            np.c_[np.log(stim[keep, :2]), np.deg2rad(stim[keep, 2]), np.log(rs[keep])]
        )
        traces_list.append(tr[keep])
        recs.append(np.full(keep.sum(), rec["name"]))
    X = np.concatenate(xs).T
    log_rs_median = np.median(X[3])
    X[3] -= log_rs_median
    return X, np.concatenate(traces_list), np.concatenate(recs), log_rs_median


def rs_tuning(X, *params, free=(), min_sigma=MIN_SIGMA):
    """grating_tuning with log SF0 / log TF0 shifted or amplitude scaled by running speed."""
    p = GratingParams(*params[:10])
    slopes = dict(zip(free, params[10:]))
    sf0 = p.sf0 + slopes.get("a_sf", 0) * X[3]
    tf0 = p.tf0 + slopes.get("a_tf", 0) * X[3]
    log_amp = p.log_amplitude + slopes.get("g", 0) * X[3]
    return grating_tuning(
        (X[0], X[1], X[2]), log_amp, sf0, tf0, *p[3:], min_sigma=min_sigma
    )


def bounds(X, free):
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
    return list(lower) + [-SLOPE_BOUND] * len(free), list(upper) + [SLOPE_BOUND] * len(
        free
    )


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


def fit_models(X, y, models=None):
    """Models on frames X, y: {model: (popt, rsq)}, running-speed models warm-started.

    `models` (default all) restricts which models are fitted; the models they are warm-started from
    are fitted too. Random starts are drawn for every model in MODELS order whether it is fitted or
    not, so a model's fit does not depend on which others are requested.
    """
    models = list(MODELS) if models is None else list(models)
    needed = set(models) | {"base"}
    for m in models:
        needed |= set(NESTED_STARTS.get(m, ()))
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
        random_starts = [
            random_p0(X, y) + list(np.random.uniform(-1, 1, len(free)))
            for _ in range(N_ITER_RS)
        ]
        if model not in needed:
            continue
        lo, hi = bounds(X, free)
        starts = [list(base) + [0.0] * len(free)]
        for simpler in NESTED_STARTS.get(model, ()):
            popt = out[simpler][0]
            if np.all(np.isfinite(popt)):
                slopes = dict(zip(MODELS[simpler], popt[10:]))
                starts.append(list(popt[:10]) + [slopes.get(t, 0.0) for t in free])
        out[model] = best_fit(
            partial(rs_tuning, free=free), X, y, lo, hi, starts + random_starts
        )
    return {m: out[m] for m in models}


def fit_roi(X, y, recording, n_jobs=1, models=None):
    """Full-data and leave-one-recording-out fits of the models (default all) for one ROI.

    The full-data fit and the folds are independent and run in parallel with `n_jobs`.
    """
    recs = np.unique(recording)
    subsets = [np.ones(len(y), bool)] + [recording != rec for rec in recs]
    if n_jobs == 1:
        fits = [fit_models(X[:, s], y[s], models) for s in subsets]
    else:
        from joblib import Parallel, delayed

        fits = Parallel(n_jobs=n_jobs)(
            delayed(fit_models)(X[:, s], y[s], models) for s in subsets
        )
    full, folds = fits[0], fits[1:]
    preds = {m: np.full(len(y), np.nan) for m in full}
    for rec, fold in zip(recs, folds):
        test = recording == rec
        for m in full:
            if np.all(np.isfinite(fold[m][0])):
                preds[m][test] = rs_tuning(X[:, test], *fold[m][0], free=MODELS[m])
    row = {}
    for m in full:
        free = MODELS[m]
        names = list(GratingParams._fields) + list(free)
        row.update({f"{m}_{n}": v for n, v in zip(names, full[m][0])})
        row[f"{m}_rsq"] = full[m][1]
        ok = np.isfinite(preds[m])
        row[f"{m}_test_rsq"] = common_utils.calculate_r_squared(y[ok], preds[m][ok])
        row[f"{m}_test_spearmanr_rval"], row[f"{m}_test_spearmanr_pval"] = spearmanr(
            y[ok], preds[m][ok]
        )
    return row


def sftf_tuned_rois(flm, session):
    """iscell ROIs that are cross-validated SF/TF tuned in the production neurons_df_sftf."""
    folder = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=flm, project=PROJECT, conflicts="skip"
    ).path_full.parent
    fit = pd.read_pickle(folder / "neurons_df_sftf.pickle")
    rois_ds = flz.get_datasets(
        origin_name=session,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        flexilims_session=flm,
        allow_multiple=False,
    )
    iscell = np.load(rois_ds.path_full / "plane0" / "iscell.npy")[:, 0] == 1
    pval = common_utils.one_sided_pval_from_spearman(
        fit.sftf_test_spearmanr_rval, fit.sftf_test_spearmanr_pval
    )
    return np.flatnonzero(iscell & (np.asarray(pval) < 0.05))


def frames_path(out_dir, session):
    return Path(out_dir) / "frames" / f"{session}.npz"


def prepare(session, out_dir, trace="spks", lag=2):
    """Sync the session once and save the running frames and SF/TF-tuned ROIs' signal."""
    flm = flz.get_flexilims_session(PROJECT)
    rois = sftf_tuned_rois(flm, session)
    X, traces, recording, log_rs_median = session_frames(
        flm, session, trace=trace, lag=lag
    )
    path = frames_path(out_dir, session)
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
    )
    print(
        f"{session} ({trace}, lag={lag}): {X.shape[1]} running frames, {len(rois)} ROIs -> {path}"
    )


def fit(session, roi, out_dir, n_jobs, models=None):
    """Fit one ROI from the prepared frames.

    Without `models`, skipped if its result exists. With `models`, those missing from an existing
    result are fitted and their columns added to it (the other columns are kept).
    """
    target = Path(out_dir) / "rois" / f"{session}_{roi}.pickle"
    old = pd.read_pickle(target) if target.exists() else None
    if models is None:
        if old is not None:
            print(f"{target} exists, skipping")
            return
    elif old is not None:
        models = [m for m in models if f"{m}_test_rsq" not in old.columns]
        if not models:
            print(f"{target} has all requested models, skipping")
            return
    d = np.load(frames_path(out_dir, session))
    (col,) = np.flatnonzero(d["rois"] == roi)
    traces = d["traces"] if "traces" in d else d["dff"]
    start = time.time()
    row = fit_roi(d["X"], traces[:, col], d["recording"], n_jobs=n_jobs, models=models)
    if old is None:
        out = pd.DataFrame([dict(session=session, roi=roi, **row)])
    else:
        out = old.assign(**row)
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
    (out_dir / f"{session}_sftf_running_models.json").write_text(
        json.dumps(
            dict(
                n_frames=int(d["X"].shape[1]),
                log_rs_median=float(d["log_rs_median"]),
                rs_thr=RS_THR,
                trace=trace,
                lag=lag,
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
        help="stimulus to trace lag in frames (prepare only, default: 2 for spks, 5 for dff)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        help="fit only these models, adding them to an existing ROI result (fit only)",
    )
    a = p.parse_args()
    if a.command == "prepare":
        lag = a.lag if a.lag is not None else (2 if a.trace == "spks" else 5)
        prepare(a.session, a.out_dir, trace=a.trace, lag=lag)
    elif a.command == "fit":
        fit(a.session, a.roi, a.out_dir, a.n_jobs, a.models)
    else:
        merge(a.session, a.out_dir)
