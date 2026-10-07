"""Snapshot RF lag sweep on the single-depth closed-loop recordings of one session.

The single-depth design of `fit_3d_rfs` puts the (lagged) stimulus of each frame in
the block of the depth shown on that frame, other blocks being zero, with the same
spatial and depth penalties as the multidepth fit. Building that as a multidepth
stimulus (one channel per depth, zero outside its trials) lets the fast search of
`fit_3d_rfs_grid_search` run the 13 x 13 hyperparameter grid for every
lag. Cross-validation splits whole trials (KFold, not stratified by depth as in
`fit_3d_rfs`), so R2 at lag 2 is close to but not exactly the pipeline value.

Usage:
    python single_depth_lag_sweep.py SESSION --out FILE --lags 0,1,2,...
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import flexiznam as flz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refit_rf_variant import PROJECT, REG_GRID, depth_channels, regenerate_single_depth  # noqa: E402

from cottage_analysis.analysis.spheres import rf_fitting  # noqa: E402
from cottage_analysis.pipelines import pipeline_utils  # noqa: E402


def main(session, out, lags, use_col="dffs"):
    fs = flz.get_flexilims_session(PROJECT)
    frames, imaging_df = regenerate_single_depth(
        fs, session, add_spikes=use_col == "spks"
    )
    frames = frames[..., int(frames.shape[-1] // 2) :].astype(np.float32)  # contra
    frame_period = np.nanmedian(np.diff(imaging_df.imaging_harptime))

    neurons_df = pd.read_pickle(
        pipeline_utils.create_neurons_ds(
            session_name=session, flexilims_session=fs, project=None, conflicts="skip"
        ).path_full
    )
    stored = np.array(
        [x[1] if np.size(x) == 2 else np.nan for x in neurons_df.rf_rsq_closedloop]
    )
    best_r2 = np.full((len(neurons_df), len(lags)), np.nan)
    best_reg = np.zeros((len(neurons_df), len(lags)), dtype=int)
    for il, lag in enumerate(lags):
        print(f"Lag {lag} frames ({lag * frame_period * 1000:.0f} ms)")
        stim = depth_channels(frames, imaging_df, lag)
        r2, reg = rf_fitting.fit_3d_rfs_grid_search(
            imaging_df.copy(),
            stim,
            reg_xys=REG_GRID,
            reg_depths=REG_GRID,
            shift_stim=0,
            use_col=use_col,
        )
        best_r2[:, il], best_reg[:, il] = r2, reg
        if lag == 2:
            ok = np.isfinite(stored)
            print(
                f"  lag 2 vs stored pipeline R2: corr {np.corrcoef(stored[ok], r2[ok])[0, 1]:.4f}, "
                f"median {np.median(stored[ok]):.4f} vs {np.median(r2[ok]):.4f}"
            )
    tmp = out + ".tmp.npz"
    np.savez(
        tmp,
        rois=neurons_df.roi.values,
        lags=np.array(lags),
        latencies=np.array(lags) * frame_period,
        best_r2=best_r2,
        best_reg=best_reg,
        stored_r2=stored,
        frame_period=frame_period,
        use_col=use_col,
    )
    os.replace(tmp, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--lags", required=True, help="comma-separated lags in imaging frames"
    )
    p.add_argument("--use-col", choices=["dffs", "spks"], default="dffs")
    a = p.parse_args()
    main(a.session, a.out, [int(x) for x in a.lags.split(",")], a.use_col)
