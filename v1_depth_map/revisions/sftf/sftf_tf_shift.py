"""Per-cell, gain-invariant test of a running-speed shift of the preferred TF.

For the exploratory grating analysis (`revisions/sftf/sftf_exploratory_analysis.ipynb`). Uses the running frames
and fits of `sftf_running_models.py` (`running_models/frames/<session>.npz` and
`running_models/<session>_sftf_running_models.pickle`, gain model included).

For each SF/TF-tuned cell:

1. the running frames of the session are split at the median running speed (slow / fast halves);
2. TF curves of the two halves over the SF / direction pairs that drive the cell (pairs >= THRESHOLD x
   the best pair over all frames, same pairs for both halves; `sftf.population_tf_curves`);
3. preferred TF of each half = centre of mass in log2 TF of (curve - its minimum)
   (`sftf_plots.tf_centre_of_mass`), which a multiplicative gain or an offset does not change;
   shift = fast - slow, in octaves;
4. null: N_SIM surrogates from the cell's fitted `gain` model (amplitude ∝ RS^g, no TF shift) plus its own
   residuals circularly shifted within each recording (same shift for all cells), analysed identically.
   This keeps the noise structure and the gain-induced SNR difference between halves; p = fraction of
   surrogates with a shift >= the observed one (one-sided, TF increasing with running speed).

Calibration (`--sham`): the same with the running-speed trace scrambled in time (circularly shifted within
each recording), so that the slow / fast split is unrelated to running; the fraction of p < 0.05 should
be ~0.05 and the same for all cell groups.

Writes RESULTS_DIR/tf_shift[_sham].pickle (one row per cell: shift, null mean / sd, p_pos) and a .json.

Usage:
    python sftf_tf_shift.py RESULTS_DIR [--n-sim 200] [--threshold 0.5] [--sham] [--seed 2]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cottage_analysis.analysis.fit_gaussian_blob import GratingParams
from v1_depth_map.figure_utils import sftf
from v1_depth_map.revisions.sftf import sftf_plots
from v1_depth_map.precompute_data.sftf_running_models import rs_tuning


def session_shifts(frames, fits, n_sim, threshold, sham, rng):
    """Observed shift, surrogate null and p for the cells of one session."""
    X, dff, rois, rec = frames["X"], frames["dff"], frames["rois"], frames["recording"]
    log_rs_median = float(frames["log_rs_median"])
    rec_idx = [np.flatnonzero(rec == r) for r in np.unique(rec)]
    X_bins = X.copy()
    if sham:  # scramble the running-speed trace in time, within recording
        for idx in rec_idx:
            X_bins[3, idx] = np.roll(
                X[3, idx], rng.integers(len(idx) // 4, 3 * len(idx) // 4)
            )
    rs_cm_s = np.exp(X_bins[3] + log_rs_median) * 100
    edges = [rs_cm_s.min() * 0.999, np.median(rs_cm_s), rs_cm_s.max() * 1.001]
    cols = [int(np.flatnonzero(rois == roi)[0]) for roi in fits.roi]
    y = dff[:, cols]
    pred = np.column_stack(
        [
            rs_tuning(
                X,
                *[row[f"gain_{p}"] for p in GratingParams._fields],
                row.gain_g,
                free=("g",),
            )
            for _, row in fits.iterrows()
        ]
    )
    resid = y - pred

    def shift_of(responses):
        curves, tf, _ = sftf.population_tf_curves(
            X_bins, responses, log_rs_median, edges, threshold=threshold
        )
        com = sftf_plots.tf_centre_of_mass(curves, tf)
        return com[:, 1] - com[:, 0]

    observed = shift_of(y)
    null = np.zeros((n_sim, len(cols)))
    for k in range(n_sim):
        shuffled = np.empty_like(resid)
        for idx in rec_idx:
            lo = len(idx) // 10
            shuffled[idx] = np.roll(resid[idx], rng.integers(lo, len(idx) - lo), axis=0)
        null[k] = shift_of(pred + shuffled)
    return fits[["session", "roi"]].assign(
        shift=observed,
        null_mean=null.mean(axis=0),
        null_sd=null.std(axis=0),
        # NaN (flat curve) shifts get no p rather than the minimum p
        p_pos=np.where(
            np.isfinite(observed),
            (1 + np.sum(null >= observed, axis=0)) / (n_sim + 1),
            np.nan,
        ),
    )


def main(results_dir, n_sim, threshold, sham, seed):
    results_dir = Path(results_dir)
    rng = np.random.default_rng(seed)
    out = []
    for path in sorted(results_dir.glob("*_sftf_running_models.pickle")):
        fits = pd.read_pickle(path)
        session = fits.session.iloc[0]
        frames = np.load(results_dir / "frames" / f"{session}.npz")
        out.append(session_shifts(frames, fits, n_sim, threshold, sham, rng))
        print(f"{session}: {len(fits)} cells", flush=True)
    out = pd.concat(out, ignore_index=True)
    name = "tf_shift_sham" if sham else "tf_shift"
    out.to_pickle(results_dir / f"{name}.pickle")
    (results_dir / f"{name}.json").write_text(
        json.dumps(
            dict(
                n_sim=n_sim,
                threshold=threshold,
                sham=sham,
                seed=seed,
                split="session median running speed",
            ),
            indent=1,
        )
    )
    print(f"Saved {len(out)} cells to {results_dir / f'{name}.pickle'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "results_dir",
        help="running_models folder (frames/ and *_sftf_running_models.pickle)",
    )
    p.add_argument("--n-sim", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--sham", action="store_true")
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()
    main(a.results_dir, a.n_sim, a.threshold, a.sham, a.seed)
