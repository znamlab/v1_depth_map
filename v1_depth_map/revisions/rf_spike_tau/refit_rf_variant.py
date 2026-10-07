"""Refit the closed-loop multidepth RFs of one session with a variant of the pipeline fit.

Runs the same steps as the RF part of the analysis pipeline (tuned contra fit,
ipsi fit at the contra hyperparameters, RF preferred depth, rf_sig) but saves the
results to a separate file and never writes neurons_df. The stimulus is the one of
the pipeline (start of each imaging frame), lagged by --shift-stim frames (2 in the
pipeline). The response is dF/F or, with --use-col spks, the suite2p spikes.

With --single-depth, the single-depth closed-loop recordings of the session are
fitted instead, with the single-depth design of `fit_3d_rfs` (stimulus in the block of
the depth shown on each frame) run through the fast multidepth search; folds split
whole trials (not stratified by depth as in `fit_3d_rfs`).

With --rs-thr, only frames with running speed above the threshold (m/s, same frame
as the response, as the `_running` depth tuning of figure 1) are fitted; trials are
still the cross-validation unit.

Usage:
    python refit_rf_variant.py SESSION --use-col spks --out DIR
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.analysis import spheres
from cottage_analysis.analysis.spheres import rf_fitting
from cottage_analysis.analysis.spheres.rf_analysis import fit_rf_preferred_depth
from cottage_analysis.pipelines import pipeline_utils

PROJECT = "colasa_3d-vision_revisions"
PROTOCOL_BASE = "SpheresPermTubeReward_multidepth"
FILTER_TRACES = {"anatomical_only": 3, "annotated": True, "ast_neuropil": False}
REG_GRID = np.geomspace(2.5, 10240, 13)
SFX = "_closedloop_multidepth"


def regenerate(flexilims_session, session, add_spikes=False):
    return spheres.regenerate_frames_all_recordings(
        session_name=session,
        flexilims_session=flexilims_session,
        project=None,
        filter_datasets=FILTER_TRACES,
        recording_type="two_photon",
        is_closedloop=1,
        is_multidepth=True,
        protocol_base=PROTOCOL_BASE,
        photodiode_protocol=5,
        return_volumes=True,
        verbose=False,
        resolution=5,
        add_spikes=add_spikes,
    )


def single_depth_trials(imaging_df):
    """Trial index as in `fit_3d_rfs`: a new trial at each change to a positive depth."""
    depth = imaging_df.depth
    trial_idx = np.cumsum((np.abs(depth.diff()) > 0) & (depth > 0)).astype(float).values
    trial_idx[~(depth > 0).values] = np.nan
    return trial_idx


def regenerate_single_depth(flexilims_session, session, add_spikes=False):
    """Single-depth closed-loop stimulus and imaging_df, with `stim` marking its trials.

    `stim` is set so that `rf_fitting._trial_index` finds the trials of `fit_3d_rfs`
    (back-to-back trials are separated by one excluded frame; a trial still running at
    the end of the recording is excluded).
    """
    frames, imaging_df = spheres.regenerate_frames_all_recordings(
        session_name=session,
        flexilims_session=flexilims_session,
        project=None,
        filter_datasets=FILTER_TRACES,
        recording_type="two_photon",
        is_closedloop=1,
        is_multidepth=False,
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=5,
        return_volumes=True,
        verbose=False,
        resolution=5,
        add_spikes=add_spikes,
    )
    ref = single_depth_trials(imaging_df)
    imaging_df["stim"] = np.isfinite(ref).astype(int)
    start = np.flatnonzero(np.diff(np.nan_to_num(ref, nan=-1)) > 0) + 1
    joined = start[np.isfinite(ref[start - 1])]
    imaging_df.loc[joined, "stim"] = 0
    md = rf_fitting._trial_index(imaging_df)
    n_ref = len(np.unique(ref[np.isfinite(ref)]))
    n_md = len(np.unique(md[np.isfinite(md)]))
    unfinished = int(np.isfinite(ref[-1]))
    assert n_md == n_ref - unfinished, f"{n_md} trials vs {n_ref} in fit_3d_rfs"
    print(
        f"{n_ref} single-depth trials ({len(joined)} back-to-back split, {unfinished} unfinished excluded)"
    )
    return frames, imaging_df


def depth_channels(frames, imaging_df, lag):
    """Stimulus lagged by `lag` frames, in the channel of the depth on the response frame.

    Returns:
        np.array: (ndepths, nframes, ele, azi), zero outside each depth's frames.
    """
    depths = np.sort(imaging_df.depth[imaging_df.depth > 0].unique())
    lagged = np.roll(frames, lag, axis=0)
    out = np.zeros((len(depths),) + frames.shape, dtype=np.float32)
    for i, d in enumerate(depths):
        m = (imaging_df.depth == d).values
        out[i, m] = lagged[m]
    return out


def replace_spikes(imaging_df, spikes_dir):
    """Put spikes from `deconvolve_tau.py --out-dir` in imaging_df.spks.

    Each recording folder in `spikes_dir` (see its paths.json) is matched to its block of imaging_df
    rows by its stored dF/F (the rows of a recording are the first frames of its
    split dff.npy), which also checks the alignment.

    Args:
        imaging_df (pd.DataFrame): concatenated imaging_df with a `dffs` column.
        spikes_dir (str): output folder of `deconvolve_tau.py --out-dir`.

    Returns:
        pd.DataFrame: imaging_df with `spks` replaced.
    """
    import json

    D = np.concatenate(imaging_df.dffs.values)
    spks = np.full(D.shape, np.nan, dtype=np.float32)
    filled = np.zeros(len(D), dtype=bool)
    paths = json.loads((Path(spikes_dir) / "paths.json").read_text())
    for name, split_path in sorted(paths.items()):
        folder = Path(spikes_dir) / name
        planes = sorted(
            folder.glob("plane*/spks.npy"), key=lambda f: int(f.parent.name[5:])
        )
        dff = np.vstack(
            [np.load(Path(split_path) / f.parent.name / "dff.npy") for f in planes]
        ).T
        new = np.vstack([np.load(f) for f in planes]).T

        def same_row(a, b):
            return np.all((a == b) | (np.isnan(a) & np.isnan(b)), axis=-1)

        hits = np.flatnonzero(same_row(D, dff[0]))
        if len(hits) == 0:
            continue  # recording not in this imaging_df (other protocol)
        start = hits[0]
        # rows of this recording: the run of rows equal to its dff (imaging_df may be truncated)
        m = min(len(dff), len(D) - start)
        same = same_row(D[start : start + m], dff[:m])
        n = m if same.all() else int(np.argmin(same))
        assert n > 0.9 * len(
            dff
        ), f"{folder.name}: only {n} of {len(dff)} frames matched"
        spks[start : start + n] = new[:n]
        filled[start : start + n] = True
        print(f"spikes from {name}: rows {start}-{start + n}")
    assert filled.all(), f"{np.sum(~filled)} imaging_df rows got no spikes"
    imaging_df["spks"] = list(spks[:, None, :])
    return imaging_df


def main(
    session,
    out_dir,
    reg_depth=None,
    rs_thr=None,
    use_col="dffs",
    spikes_dir=None,
    shift_stim=2,
    single_depth=False,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = (
        out_dir / f"{session}_rf_{'single_depth' if single_depth else 'snapshot'}.pkl"
    )
    if target.exists():
        print(f"{target} exists, skipping")
        return

    flexilims_session = flz.get_flexilims_session(PROJECT)
    if single_depth:
        frames, imaging_df = regenerate_single_depth(
            flexilims_session, session, add_spikes=use_col == "spks"
        )
    else:
        frames, imaging_df = regenerate(
            flexilims_session, session, add_spikes=use_col == "spks"
        )
    if spikes_dir is not None:
        imaging_df = replace_spikes(imaging_df, spikes_dir)
    half = int(frames.shape[-1] // 2)
    contra, ipsi = frames[..., half:], frames[..., :half]
    sfx = "_closedloop" if single_depth else SFX
    lag = shift_stim
    if single_depth:
        # lag applied when building the depth channels
        contra = depth_channels(contra, imaging_df, shift_stim)
        ipsi = depth_channels(ipsi, imaging_df, shift_stim)
        shift_stim = 0
    if rs_thr is not None:
        in_trial = np.isfinite(rf_fitting._trial_index(imaging_df))
        imaging_df["rf_use_frame"] = (imaging_df.RS > rs_thr).values
        print(
            f"RS > {rs_thr} m/s: {imaging_df.rf_use_frame[in_trial].mean():.1%} of in-trial frames kept"
        )
    grid = REG_GRID
    depth_grid = grid if reg_depth is None else np.array([reg_depth])

    coef, r2, best_reg_xys, best_reg_depths = rf_fitting.fit_3d_rfs_hyperparam_tuning(
        imaging_df,
        contra,
        reg_xys=grid,
        reg_depths=depth_grid,
        shift_stim=shift_stim,
        use_col=use_col,
        k_folds=5,
    )
    coef_ipsi, r2_ipsi = rf_fitting.fit_3d_rfs_ipsi(
        imaging_df,
        ipsi,
        best_reg_xys,
        best_reg_depths,
        shift_stim=shift_stim,
        use_col=use_col,
        k_folds=5,
    )

    # Fill a copy of neurons_df as the pipeline does, without saving it
    neurons_df = pd.read_pickle(
        pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        ).path_full
    )
    assert len(neurons_df) == coef.shape[2]
    neurons_df[f"rf_coef{sfx}"] = [coef[:, :, i].copy() for i in range(coef.shape[2])]
    depth_list = np.sort(imaging_df.depth[imaging_df.depth > 0].unique()).tolist()
    assert len(depth_list) == contra.shape[0]
    fit_rf_preferred_depth(
        neurons_df, depths=depth_list, is_closed_loop=1, use_multidepth=not single_depth
    )
    rf_sig, rf_sig_ipsi = rf_fitting.find_sig_rfs(list(coef), list(coef_ipsi), n_std=6)

    out = pd.DataFrame(
        {
            "session": session,
            "roi": neurons_df["roi"].values,
            "rf_sig": rf_sig,
            "rf_sig_ipsi": rf_sig_ipsi,
            "rf_rsq_test": r2[:, -1],
            "rf_rsq_ipsi_test": r2_ipsi[:, -1],
            "rf_reg_xy": best_reg_xys,
            "rf_reg_depth": best_reg_depths,
            "rf_preferred_depth": neurons_df[f"rf_preferred_depth{sfx}"].values,
            "rf_depth_rsq": neurons_df[f"rf_depth_rsq{sfx}"].values,
            # RF depth Gaussian in ln(depth): sigma^2 = exp(p[2]) + min_sigma (0.5)
            "rf_depth_sigma": [
                np.sqrt(np.exp(p[2]) + 0.5) if np.size(p) == 4 else np.nan
                for p in neurons_df[f"rf_depth_popt{sfx}"].values
            ],
        }
    )
    params = dict(
        reg_grid=grid,
        reg_depth_grid=depth_grid,
        rs_thr=rs_thr,
        use_col=use_col,
        spikes_dir=spikes_dir,
        shift_stim=lag,
        single_depth=single_depth,
    )
    tmp = target.with_suffix(".tmp.pkl")
    pd.to_pickle(
        {
            "table": out,
            "coef_mean": np.nanmean(coef, axis=0).T.astype(np.float32),
            "params": params,
        },
        tmp,
    )
    os.replace(tmp, target)
    print(f"Saved {target}: rf_sig {rf_sig.mean():.1%}, params {params}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--reg-depth",
        type=float,
        default=None,
        help="fixed depth regularisation (e.g. 0) instead of the grid",
    )
    parser.add_argument(
        "--rs-thr",
        type=float,
        default=None,
        help="fit only frames with running speed above this (m/s), e.g. 0.05",
    )
    parser.add_argument(
        "--use-col",
        choices=["dffs", "spks"],
        default="dffs",
        help="response: dF/F or the suite2p spikes",
    )
    parser.add_argument(
        "--spikes-dir",
        default=None,
        help="spikes from deconvolve_tau.py --out-dir instead of the suite2p ones",
    )
    parser.add_argument(
        "--shift-stim", type=int, default=2, help="snapshot lag in frames"
    )
    parser.add_argument(
        "--single-depth", action="store_true", help="fit the single-depth recordings"
    )
    args = parser.parse_args()
    main(
        args.session,
        args.out,
        reg_depth=args.reg_depth,
        rs_thr=args.rs_thr,
        use_col=args.use_col,
        spikes_dir=args.spikes_dir,
        shift_stim=args.shift_stim,
        single_depth=args.single_depth,
    )
