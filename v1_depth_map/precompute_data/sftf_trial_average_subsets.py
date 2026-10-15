"""Fit the SFTF base model by trial average across running-speed subsets.

For the reviewer grating analysis (evaluating state-dependent tuning without parametric speed models).
Extracts grating trials and fits the 10-parameter base model (`fit_gaussian_blob.grating_tuning`)
separately on three subsets of imaging frames:

    1. static:   RS < 0.05 m/s (5 cm/s) in the frame and for the preceding 1 s (15 frames)
    2. run_slow: 5 cm/s <= RS < 50 cm/s (0.05 <= RS < 0.50 m/s)
    3. run_fast: RS >= 50 cm/s (RS >= 0.50 m/s)

Within each trial, neural responses are averaged only over frames belonging to the subset. The trial window
skips the first SKIP volumes after the grating onset and ends at the next onset (as
`gratings.generate_trials_df`); each response frame is paired with the RS (and still criterion) `lag` frames
earlier.
Trials with fewer than `min_valid_frames` (default 2) for a given subset are excluded from that subset.

Each ROI is fitted on all valid trials (parameters, `rsq`) and evaluated with 5-fold leave-one-recording-out
cross-validation (`sftf_test_rsq`, `sftf_test_spearmanr_rval`, `sftf_test_spearmanr_pval`).
A cell is considered tuned if one-sided Spearman p < 0.05.

Usage:
    python sftf_trial_average_subsets.py {prepare,fit,run,merge,summary} SESSION
        --out-dir DIR [--n-jobs 1] [--trace {dff,spks}] [--lag LAG] [--min-frames 2]
"""

import argparse
import time
from functools import partial
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cottage_analysis.analysis import common_utils
from cottage_analysis.analysis.fit_gaussian_blob import (
    GratingParams,
    grating_tuning,
    _sftf_p0_func,
    _sftf_folds,
)
from cottage_analysis.preprocessing import synchronisation

PROJECT = "hey2_3d-vision_foodres_20220101"
PHOTODIODE_PROTOCOL = 5
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}
PARAMS = ["SpatialFrequency", "TemporalFrequency", "Angle"]
STILL_THR = 0.05  # m/s (5 cm/s), max RS of still frames, matching Figure 1
STILL_FRAMES = 15  # frames t - 14 (~1 s at 15 Hz) to t + lag
RUN_SLOW_RANGE = (0.05, 0.50)  # 5 to 50 cm/s (m/s)
RUN_FAST_MIN = 0.50  # >= 50 cm/s (m/s)
DEFAULT_MIN_FRAMES = 2
SKIP = 8  # onset volumes skipped in each trial (0.53 s), see revisions/sftf/sftf_onset_profile.ipynb
MIN_SIGMA = 0.25
N_ITER = 10

SESSIONS = [
    "PZAH10.2d_S20230920",
    "PZAH10.2d_S20230922",
    "PZAH10.2f_S20230817",
    "PZAH10.2f_S20230912",
    "PZAH10.2f_S20230914",
    "PZAH10.2f_S20230924",
]

SUBSETS = ["static", "run_slow", "run_fast"]


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


def prepare_session_trials(
    flm, session, trace="dff", lag=None, min_frames=DEFAULT_MIN_FRAMES
):
    """Extract and trial-average responses for all three running-speed subsets.

    Returns:
        dict: {
            "subsets": {
                subset_name: {
                    "trials_df": pd.DataFrame (SF, TF, Angle, recording, mean_rs, n_frames),
                    "responses": np.ndarray (n_valid_trials, n_rois)
                }
            },
            "rois": np.ndarray of ROI indices,
            "iscell": np.ndarray of bool,
            "session": session,
            "trace": trace,
            "lag": lag
        }
    """
    if lag is None:
        lag = 0 if trace == "dff" else 2

    iscell = iscell_rois(flm, session)
    n_total_rois = len(iscell)

    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()

    subset_trials = {s: [] for s in SUBSETS}
    subset_resps = {s: [] for s in SUBSETS}

    for _, rec in recordings.iterrows():
        vs_df = synchronisation.generate_vs_df(
            recording=rec,
            photodiode_protocol=PHOTODIODE_PROTOCOL,
            flexilims_session=flm,
            project=PROJECT,
        )
        img_df = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=rec,
            flexilims_session=flm,
            filter_datasets=FILTER_TRACES,
            return_volumes=True,
            add_spikes=(trace == "spks"),
        ).reset_index(drop=True)

        rs = (img_df.mouse_z_harp.diff() / img_df.mouse_z_harptime.diff()).to_numpy()
        tr = np.vstack(img_df.spks.values if trace == "spks" else img_df.dffs.values)

        is_still = np.zeros(len(rs), bool)
        is_still[
            common_utils.find_thresh_sequence(
                array=rs,
                threshold_max=STILL_THR,
                length=STILL_FRAMES + lag,
                shift=STILL_FRAMES,
            )
        ] = True

        stim = img_df["stimulus_harptime"]
        onset = (stim.notna() & (stim != stim.shift())).values
        starts = np.flatnonzero(onset)
        if len(starts) == 0:
            continue
        ends = np.append(starts[1:], starts[-1] + int(np.median(np.diff(starts))))
        starts, ends = starts + SKIP, np.minimum(ends, len(img_df))

        rec_trials_meta = img_df.loc[onset, PARAMS].reset_index(drop=True)

        for i, (s, e) in enumerate(zip(starts, ends)):
            if e <= s:
                continue
            resp_idx = np.arange(max(s, lag), e)  # response frames
            trial_rs = rs[resp_idx - lag]  # RS of the frames they are paired with
            trial_tr = tr[resp_idx]
            trial_still = is_still[resp_idx - lag]

            # 1. Static
            mask_static = trial_still
            # 2. Moderate running (5 to 50 cm/s)
            mask_slow = (trial_rs >= RUN_SLOW_RANGE[0]) & (trial_rs < RUN_SLOW_RANGE[1])
            # 3. Fast running (>= 50 cm/s)
            mask_fast = trial_rs >= RUN_FAST_MIN

            masks = {
                "static": mask_static,
                "run_slow": mask_slow,
                "run_fast": mask_fast,
            }

            for sname, mask in masks.items():
                if np.sum(mask) >= min_frames:
                    row = rec_trials_meta.iloc[i].to_dict()
                    row["recording"] = rec["name"]
                    row["trial_idx"] = i
                    row["n_frames"] = int(np.sum(mask))
                    row["mean_rs"] = float(np.nanmean(trial_rs[mask]))
                    subset_trials[sname].append(row)
                    subset_resps[sname].append(np.mean(trial_tr[mask], axis=0))

    out_subsets = {}
    for sname in SUBSETS:
        if len(subset_trials[sname]) > 0:
            df = pd.DataFrame(subset_trials[sname])
            resps = np.vstack(subset_resps[sname])
        else:
            df = pd.DataFrame(
                columns=PARAMS + ["recording", "trial_idx", "n_frames", "mean_rs"]
            )
            resps = np.empty((0, n_total_rois))
        out_subsets[sname] = {"trials_df": df, "responses": resps}
        print(f"  {session} [{sname}]: {len(df)} trials with >={min_frames} frames")

    return {
        "subsets": out_subsets,
        "rois": np.arange(n_total_rois),
        "iscell": iscell,
        "session": session,
        "trace": trace,
        "lag": lag,
        "skip": SKIP,
        "min_frames": min_frames,
    }


def fit_roi_base(
    X, y, lower_bounds, upper_bounds, folds, niter=N_ITER, min_sigma=MIN_SIGMA
):
    """Fit base grating model to one ROI with cross-validation across folds."""
    func = partial(grating_tuning, min_sigma=min_sigma)
    popt, rsq = common_utils.iterate_fit(
        func,
        X,
        y,
        lower_bounds,
        upper_bounds,
        niter=niter,
        p0_func=_sftf_p0_func(X, y),
        verbose=False,
    )
    out = dict(GratingParams(*popt)._asdict(), rsq=rsq)
    if not folds:
        return out

    y_pred = np.full(len(y), np.nan)
    popts = []
    for train, test in folds:
        popt_fold, _ = common_utils.iterate_fit(
            func,
            X[:, train],
            y[train],
            lower_bounds,
            upper_bounds,
            niter=niter,
            p0_func=_sftf_p0_func(X[:, train], y[train]),
            verbose=False,
        )
        y_pred[test] = func(X[:, test], *popt_fold)
        popts.append(popt_fold)

    valid = np.isfinite(y) & np.isfinite(y_pred)
    if np.sum(valid) >= 3:
        rval, pval = spearmanr(y[valid], y_pred[valid])
        test_rsq = common_utils.calculate_r_squared(y[valid], y_pred[valid])
    else:
        rval, pval, test_rsq = np.nan, np.nan, np.nan

    out["sftf_test_rsq"] = test_rsq
    out["sftf_test_spearmanr_rval"] = rval
    out["sftf_test_spearmanr_pval"] = pval
    out["sftf_test_popts"] = popts
    return out


def fit_subset_data(
    trials_df, responses, rois_to_fit, n_jobs=1, niter=N_ITER, k_folds=5
):
    """Fit all requested ROIs on a subset's trial-averaged responses."""
    if len(trials_df) < 10:
        print(f"Warning: only {len(trials_df)} trials, skipping fit")
        return pd.DataFrame()

    log_sf = np.log(trials_df["SpatialFrequency"].to_numpy(float))
    log_tf = np.log(trials_df["TemporalFrequency"].to_numpy(float))
    angle_rad = np.deg2rad(trials_df["Angle"].to_numpy(float))
    X = np.vstack([log_sf, log_tf, angle_rad])

    lower_bounds = GratingParams(
        log_amplitude=-np.inf,
        sf0=log_sf.min() - 1,
        tf0=log_tf.min() - 1,
        log_sigma_x2=-np.inf,
        log_sigma_y2=-np.inf,
        theta=0,
        offset=-np.inf,
        alpha0=0,
        log_kappa=-np.inf,
        dsi=0,
    )
    upper_bounds = GratingParams(
        log_amplitude=np.inf,
        sf0=log_sf.max() + 1,
        tf0=log_tf.max() + 1,
        log_sigma_x2=np.inf,
        log_sigma_y2=np.inf,
        theta=0.5 * np.pi,
        offset=np.inf,
        alpha0=2 * np.pi,
        log_kappa=6.0,  # MAX_LOG_KAPPA
        dsi=1,
    )

    df_folds = trials_df[
        ["SpatialFrequency", "TemporalFrequency", "Angle", "recording"]
    ].copy()
    folds = (
        _sftf_folds(df_folds, k_folds, fold_col="recording") if k_folds > 1 else None
    )

    fit_fn = partial(
        fit_roi_base,
        X=X,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        folds=folds,
        niter=niter,
        min_sigma=MIN_SIGMA,
    )

    if n_jobs == 1:
        rows = [fit_fn(y=responses[:, roi]) for roi in rois_to_fit]
    else:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fit_fn)(y=responses[:, roi]) for roi in rois_to_fit
        )

    out = pd.DataFrame(rows)
    out.insert(0, "roi", rois_to_fit)
    out["n_valid_trials"] = len(trials_df)
    out["mean_rs_cm_s"] = float(trials_df["mean_rs"].mean() * 100)
    out["pref_speed"] = np.exp(out["tf0"] - out["sf0"])
    out["pref_ori"] = np.rad2deg(out["alpha0"]) % 180
    one_sided_p = common_utils.one_sided_pval_from_spearman(
        out["sftf_test_spearmanr_rval"], out["sftf_test_spearmanr_pval"]
    )
    out["sftf_test_spearmanr_pval_one_sided"] = one_sided_p
    out["sftf_tuned"] = one_sided_p < 0.05
    return out


def prepared_path(out_dir, session):
    return Path(out_dir) / "prepared" / f"{session}_prepared_trials.pickle"


def prepare(session, out_dir, trace="dff", lag=None, min_frames=DEFAULT_MIN_FRAMES):
    flm = flz.get_flexilims_session(PROJECT)
    start = time.time()
    data = prepare_session_trials(
        flm, session, trace=trace, lag=lag, min_frames=min_frames
    )
    target = prepared_path(out_dir, session)
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(data, target)
    print(
        f"Prepared trials for {session} saved to {target} in {time.time() - start:.1f} s"
    )


def fit(session, out_dir, n_jobs=1, iscell_only=True, niter=N_ITER):
    data_path = prepared_path(out_dir, session)
    assert data_path.exists(), f"Prepared data missing: {data_path}. Run prepare first."
    data = pd.read_pickle(data_path)

    rois = data["rois"]
    iscell = data["iscell"]
    rois_to_fit = rois[iscell] if iscell_only else rois
    print(f"Fitting {len(rois_to_fit)} ROIs for {session} across {SUBSETS}...")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_df = pd.DataFrame(
        {"session": session, "roi": rois_to_fit, "iscell": iscell[rois_to_fit]}
    )

    for sname in SUBSETS:
        sub_dict = data["subsets"][sname]
        trials_df = sub_dict["trials_df"]
        responses = sub_dict["responses"]

        t0 = time.time()
        fit_df = fit_subset_data(
            trials_df, responses, rois_to_fit, n_jobs=n_jobs, niter=niter
        )
        if len(fit_df) > 0:
            fit_df.insert(0, "session", session)
            fit_df["iscell"] = iscell[rois_to_fit]
            fit_target = out_dir / f"{session}_sftf_trial_avg_{sname}.pickle"
            fit_df.to_pickle(fit_target)
            print(
                f"  [{sname}] fitted {len(fit_df)} ROIs ({fit_df.sftf_tuned.sum()} tuned) in {time.time() - t0:.1f} s -> {fit_target.name}"
            )

            # Add to combined table with prefix
            cols_to_add = {
                c: f"{sname}_{c}"
                for c in fit_df.columns
                if c not in ("session", "roi", "iscell", "sftf_test_popts")
            }
            sub_renamed = fit_df[["roi"] + list(cols_to_add.keys())].rename(
                columns=cols_to_add
            )
            combined_df = combined_df.merge(sub_renamed, on="roi", how="left")

    combined_target = out_dir / f"{session}_sftf_trial_avg_combined.pickle"
    combined_df.to_pickle(combined_target)
    print(f"Combined results saved -> {combined_target}")


def run_session(
    session,
    out_dir,
    n_jobs=1,
    trace="dff",
    lag=None,
    min_frames=DEFAULT_MIN_FRAMES,
    iscell_only=True,
):
    prepare(session, out_dir, trace=trace, lag=lag, min_frames=min_frames)
    fit(session, out_dir, n_jobs=n_jobs, iscell_only=iscell_only)


def summary(out_dir):
    out_dir = Path(out_dir)
    combined_files = sorted(
        [
            f
            for f in out_dir.glob("*_sftf_trial_avg_combined.pickle")
            if not f.name.startswith(".")
        ]
    )
    if not combined_files:
        print(f"No combined pickle files found in {out_dir}")
        return

    dfs = [pd.read_pickle(f) for f in combined_files]
    all_df = pd.concat(dfs, ignore_index=True)
    out_path = out_dir / "sftf_trial_avg_combined_all_sessions.pickle"
    all_df.to_pickle(out_path)
    print(f"Aggregated {len(all_df)} ROIs across {len(dfs)} sessions -> {out_path}")

    # Summary table
    print("\nTuned cell counts across subsets:")
    for sname in SUBSETS:
        col = f"{sname}_sftf_tuned"
        if col in all_df:
            tuned = all_df[col].sum()
            print(
                f"  {sname:10s}: {tuned:4d} / {len(all_df)} ({tuned / len(all_df) * 100:.1f}%)"
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["prepare", "fit", "run", "summary"])
    p.add_argument("session", nargs="?", help="Session name (or omit for summary)")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel CPU workers")
    p.add_argument("--trace", choices=["dff", "spks"], default="dff")
    p.add_argument("--lag", type=int, default=None)
    p.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    p.add_argument(
        "--all-rois", action="store_true", help="Fit all ROIs, not only iscell"
    )

    a = p.parse_args()

    if a.command == "prepare":
        assert a.session, "Session required for prepare"
        prepare(a.session, a.out_dir, trace=a.trace, lag=a.lag, min_frames=a.min_frames)
    elif a.command == "fit":
        assert a.session, "Session required for fit"
        fit(a.session, a.out_dir, n_jobs=a.n_jobs, iscell_only=not a.all_rois)
    elif a.command == "run":
        assert a.session, "Session required for run"
        run_session(
            a.session,
            a.out_dir,
            n_jobs=a.n_jobs,
            trace=a.trace,
            lag=a.lag,
            min_frames=a.min_frames,
            iscell_only=not a.all_rois,
        )
    elif a.command == "summary":
        summary(a.out_dir)
