"""Does running speed shift the preferred SF / TF of grating responses? Frame-level model comparison.

For the reviewer grating analysis (`figures/fig_reviewer_gratings.ipynb`). Fits four versions of the
SF/TF/direction model of `fit_gaussian_blob.grating_tuning` to single imaging frames of the SFTF
recordings, with the preferred log SF and log TF allowed to depend linearly on log running speed:

    log SF0 = sf0 + a_sf * r,   log TF0 = tf0 + a_tf * r,   r = log(RS) - median log(RS) of the session

    base: a_sf = a_tf = 0;  tf: a_sf = 0;  sf: a_tf = 0;  both: free

Samples are imaging frames during gratings with RS > RS_THR (static frames excluded), dF/F shifted by
LAG frames relative to the stimulus and RS. Each model is fitted on all frames (parameters, `rsq`) and
cross-validated leaving one recording out (held-out R2 and Spearman of the concatenated predictions).
The running-speed models start from the base fit with zero slopes, plus random starts.

Only `iscell` ROIs whose production `neurons_df_sftf` fit is cross-validated SF/TF tuned (one-sided
Spearman p < 0.05) are fitted. Production files are only read. Three steps, so that ROIs can be fitted
as separate jobs:

    prepare SESSION      sync once, save running frames and dF/F to OUT_DIR/frames/<session>.npz
    fit SESSION ROI      fit one ROI -> OUT_DIR/rois/<session>_<roi>.pickle (skipped if it exists)
    merge SESSION        -> OUT_DIR/<session>_sftf_running_models.pickle (one row per ROI) + .json

Usage:
    python sftf_running_models.py {prepare,fit,merge} SESSION [ROI] --out-dir DIR [--n-jobs 1]
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
LAG = 5  # frames between stimulus / RS and dF/F (best dF/F lag for GCaMP6f in the RS/OF lag sweep)
MIN_SIGMA = 0.25
N_ITER_BASE = 5
N_ITER_RS = 3  # random starts of the running-speed models, on top of the base-fit start
SLOPE_BOUND = 5  # |a_sf|, |a_tf| in log SF (TF) per log RS
MODELS = {
    "base": (),
    "gain": ("g",),
    "tf": ("a_tf",),
    "gain_tf": ("g", "a_tf"),
    "sf": ("a_sf",),
    "both": ("a_sf", "a_tf"),
}


def session_frames(flm, session):
    """Running frames during gratings of all SFTF recordings of a session.

    Returns:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - median log RS;
        dff (n_frames, n_rois), recording (n_frames,), median log RS.
    """
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()
    xs, dffs, recs = [], [], []
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
        ).reset_index(drop=True)
        rs = (img.mouse_z_harp.diff() / img.mouse_z_harptime.diff()).to_numpy()
        stim = img[PARAMS].to_numpy(float)
        dff = np.vstack(img.dffs.values)
        # stimulus and RS of frame t with dF/F of frame t + LAG
        stim, rs, dff = stim[:-LAG], rs[:-LAG], dff[LAG:]
        keep = np.isfinite(stim).all(axis=1) & np.isfinite(rs) & (rs > RS_THR)
        xs.append(
            np.c_[np.log(stim[keep, :2]), np.deg2rad(stim[keep, 2]), np.log(rs[keep])]
        )
        dffs.append(dff[keep])
        recs.append(np.full(keep.sum(), rec["name"]))
    X = np.concatenate(xs).T
    log_rs_median = np.median(X[3])
    X[3] -= log_rs_median
    return X, np.concatenate(dffs), np.concatenate(recs), log_rs_median


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


def fit_models(X, y):
    """All models on frames X, y: {model: (popt, rsq)}, running-speed models warm-started."""
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
        lo, hi = bounds(X, free)
        starts = [list(base) + [0.0] * len(free)]
        if model == "gain_tf" and "gain" in out and "tf" in out:
            if np.all(np.isfinite(out["gain"][0])):
                starts.append(list(out["gain"][0][:10]) + [out["gain"][0][10], 0.0])
            if np.all(np.isfinite(out["tf"][0])):
                starts.append(list(out["tf"][0][:10]) + [0.0, out["tf"][0][10]])
        starts += [
            random_p0(X, y) + list(np.random.uniform(-1, 1, len(free)))
            for _ in range(N_ITER_RS)
        ]
        out[model] = best_fit(partial(rs_tuning, free=free), X, y, lo, hi, starts)
    return out


def fit_roi(X, y, recording, n_jobs=1):
    """Full-data and leave-one-recording-out fits of the four models for one ROI.

    The full-data fit and the folds are independent and run in parallel with `n_jobs`.
    """
    recs = np.unique(recording)
    subsets = [np.ones(len(y), bool)] + [recording != rec for rec in recs]
    if n_jobs == 1:
        fits = [fit_models(X[:, s], y[s]) for s in subsets]
    else:
        from joblib import Parallel, delayed

        fits = Parallel(n_jobs=n_jobs)(
            delayed(fit_models)(X[:, s], y[s]) for s in subsets
        )
    full, folds = fits[0], fits[1:]
    preds = {m: np.full(len(y), np.nan) for m in MODELS}
    for rec, fold in zip(recs, folds):
        test = recording == rec
        for m, free in MODELS.items():
            if np.all(np.isfinite(fold[m][0])):
                preds[m][test] = rs_tuning(X[:, test], *fold[m][0], free=free)
    row = {}
    for m, free in MODELS.items():
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


def prepare(session, out_dir):
    """Sync the session once and save the running frames and SF/TF-tuned ROIs' dF/F."""
    flm = flz.get_flexilims_session(PROJECT)
    rois = sftf_tuned_rois(flm, session)
    X, dff, recording, log_rs_median = session_frames(flm, session)
    path = frames_path(out_dir, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=X,
        dff=dff[:, rois],
        rois=rois,
        recording=recording,
        log_rs_median=log_rs_median,
    )
    print(f"{session}: {X.shape[1]} running frames, {len(rois)} ROIs -> {path}")


def fit(session, roi, out_dir, n_jobs):
    """Fit one ROI from the prepared frames; skipped if its result exists."""
    target = Path(out_dir) / "rois" / f"{session}_{roi}.pickle"
    if target.exists():
        print(f"{target} exists, skipping")
        return
    d = np.load(frames_path(out_dir, session))
    (col,) = np.flatnonzero(d["rois"] == roi)
    start = time.time()
    row = fit_roi(d["X"], d["dff"][:, col], d["recording"], n_jobs=n_jobs)
    row = dict(session=session, roi=roi, **row)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_pickle(target)
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
    (out_dir / f"{session}_sftf_running_models.json").write_text(
        json.dumps(
            dict(
                n_frames=int(d["X"].shape[1]),
                log_rs_median=float(d["log_rs_median"]),
                rs_thr=RS_THR,
                lag=LAG,
                filter_traces=FILTER_TRACES,
                n_iter_base=N_ITER_BASE,
                n_iter_rs=N_ITER_RS,
                slope_bound=SLOPE_BOUND,
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
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.session, a.out_dir)
    elif a.command == "fit":
        fit(a.session, a.roi, a.out_dir, a.n_jobs)
    else:
        merge(a.session, a.out_dir)
