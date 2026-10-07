"""Sweep the stimulus lag of the multidepth RF fit for well-fitted cells of one session.

The stimulus at the start of each imaging frame (as in the pipeline) is shifted by
each lag, in whole frames, and the RF of the selected cells is refitted with the same
hyperparameter search as the pipeline, giving the best cross-validated R2 per cell
and lag. RF coefficients of a few example cells are also saved at each lag (at their
own fixed hyperparameters) to make movies.

Cells are selected from a previous fit (`refit_rf_variant.py` output) by its R2.

Usage:
    python latency_sweep.py SESSION --fit FILE --lags 0,1,2,3,4,5,6 --out FILE [--use-col spks]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import flexiznam as flz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refit_rf_variant import PROJECT, REG_GRID, regenerate, replace_spikes  # noqa: E402

from cottage_analysis.analysis.spheres import rf_fitting  # noqa: E402

DEPTHS_CM = np.array([5, 10, 20, 40, 80, 160, 320, 640])


def pick_examples(table, coef_mean, n_per_group=2):
    """Best-fitted cells with RF peaks at near, middle and far depths."""
    peak = np.argmax(coef_mean[:, :-1].reshape(len(table), 8, -1).max(axis=2), axis=1)
    groups = {"near": peak <= 2, "middle": (peak >= 3) & (peak <= 5), "far": peak >= 6}
    picks = []
    for mask in groups.values():
        cand = table[mask & table.rf_sig.values].sort_values(
            "rf_rsq_test", ascending=False
        )
        picks += cand.index[:n_per_group].tolist()
    return np.array(picks)


def main(
    session, fit_file, out, min_r2, lags, examples=True, use_col="dffs", spikes_dir=None
):
    fit = pd.read_pickle(fit_file)
    table = fit["table"]
    rois = np.flatnonzero(table.rf_rsq_test.values > min_r2)
    examples = (
        pick_examples(table, fit["coef_mean"]) if examples else np.array([], dtype=int)
    )
    print(f"{len(rois)} cells with R2 > {min_r2}; example cells {examples.tolist()}")

    flexilims_session = flz.get_flexilims_session(PROJECT)
    frames, imaging_df = regenerate(
        flexilims_session, session, add_spikes=use_col == "spks"
    )
    if spikes_dir is not None:
        imaging_df = replace_spikes(imaging_df, spikes_dir)
    frame_period = np.nanmedian(np.diff(imaging_df.imaging_harptime))
    stim = frames[..., int(frames.shape[-1] // 2) :].astype(np.float32)  # contra
    del frames

    grid = np.array([[a, b] for a in REG_GRID for b in REG_GRID])
    best_r2 = np.full((len(rois), len(lags)), np.nan)
    best_reg = np.zeros((len(rois), len(lags)), dtype=int)
    movie = np.zeros((len(examples), len(lags), 8, 16, 24), dtype=np.float32)
    for il, lag in enumerate(lags):
        print(f"Lag {lag} frames ({lag * frame_period * 1000:.0f} ms)")
        r2, reg = rf_fitting.fit_3d_rfs_grid_search(
            imaging_df.copy(),
            stim,
            reg_xys=REG_GRID,
            reg_depths=REG_GRID,
            shift_stim=lag,
            use_col=use_col,
            choose_rois=rois,
        )
        best_r2[:, il], best_reg[:, il] = r2, reg
        # example RFs at the hyperparameters of the selection fit, fixed across lags
        for ie, roi in enumerate(examples):
            coef, _ = rf_fitting.fit_3d_rfs(
                imaging_df.copy(),
                stim,
                reg_xy=table.rf_reg_xy[roi],
                reg_depth=table.rf_reg_depth[roi],
                shift_stim=lag,
                use_col=use_col,
                choose_rois=[roi],
            )
            movie[ie, il] = np.mean(coef, axis=0)[:-1, 0].reshape(8, 16, 24)

    tmp = out + ".tmp.npz"
    np.savez(
        tmp,
        rois=rois,
        lags=np.array(lags),
        latencies=np.array(lags) * frame_period,
        best_r2=best_r2,
        best_reg=best_reg,
        reg_grid=grid,
        examples=examples,
        movie=movie,
        depths_cm=DEPTHS_CM,
        frame_period=frame_period,
        use_col=use_col,
        spikes_dir=str(spikes_dir),
    )
    os.replace(tmp, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session")
    parser.add_argument(
        "--fit", required=True, help="refit_rf_variant.py output used to select cells"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-r2", type=float, default=0.05)
    parser.add_argument(
        "--lags", default="0,1,2,3,4,5,6", help="comma-separated lags in imaging frames"
    )
    parser.add_argument(
        "--no-examples", action="store_true", help="skip the example RF movies"
    )
    parser.add_argument("--use-col", choices=["dffs", "spks"], default="dffs")
    parser.add_argument(
        "--spikes-dir",
        default=None,
        help="spikes from deconvolve_tau.py --out-dir instead of the suite2p ones",
    )
    args = parser.parse_args()
    main(
        args.session,
        args.fit,
        args.out,
        args.min_r2,
        [int(x) for x in args.lags.split(",")],
        examples=not args.no_examples,
        use_col=args.use_col,
        spikes_dir=args.spikes_dir,
    )
