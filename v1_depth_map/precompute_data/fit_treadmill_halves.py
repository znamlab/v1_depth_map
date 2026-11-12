"""Fig 3 K-M control: treadmill RS/OF fits on the first or second half of each trial.

Same fit as the `treadmill` target of `fit_revision_treadmill.py` that feeds Fig 3 K-M
(`_treadmill_trial_average_plateau`): onset by the `plateau` ramp detector plus the 0.5 s
margin, trial averages, `max_rs2motor_diff=0.3`, `TREADMILL_PARAM_RANGE`, legacy g2d
parameterisation. The only change: after loading, every per-frame `*_stim` array of each
trial is cut to its first or second half, so a residual response to the acceleration ramp
would show up as a difference between the two halves.

- `half1` keeps frames `[0, n // 2)` of the analysis window, `half2` keeps
  `[n - n // 2, n)`; the middle frame of odd-length trials is dropped so both halves have
  the same length.
- The RS-to-motor filter (`max_abs_rs2motor_diff_ratio`, a 0.5 s trailing window) is
  computed on the continuous trace before the split, so each frame keeps the mask it has in
  the production fit; it is applied per half inside `fit_rs_of_tuning`.
- `full` applies no split: a check that this code reproduces the production pickles.

Outputs (never written to the session folders or neurons_df):
    OUT_DIR/SESSION/fit_rs_of_tuning_<model>[_crossval]_k<n>_treadmill_trial_average_legacy_plateau_<variant>.pickle
plus a `.json` provenance record per (session, variant).

Usage:
    python fit_treadmill_halves.py SESSION VARIANT --out-dir DIR [--configs gaussian_2d:None:5 ...]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cottage_analysis.analysis import fit_gaussian_blob
from cottage_analysis.pipelines import pipeline_utils
from v1_depth_map.precompute_data.fit_revision_treadmill import (
    COMMON_PARAMS,
    FILTER_DATASETS,
    PHOTODIODE_PROTOCOL,
    PROJECT,
    _cottage_analysis_sha,
    fit_filename,
    parse_config,
    report_rsof_range,
    target_config,
)

TARGET, METHOD = "treadmill", "plateau"
VARIANTS = ("half1", "half2", "full")
# Fig 3 K-M read the g2d fit: popt from k1, test R^2 (significance) from k5.
DEFAULT_CONFIGS = [("gaussian_2d", None, 1), ("gaussian_2d", None, 5)]


def split_trials(trials_df, variant):
    """Cut every per-frame `*_stim` array of each trial to one half of the window.

    Per-frame columns are found by length: any `*_stim` entry whose first dimension equals
    the trial's `RS_stim` length. Scalar and `_blank` columns are left untouched.
    Also updates `imaging_stim_start`/`imaging_stim_stop` to the kept frames.
    """
    if variant == "full":
        return trials_df
    trials_df = trials_df.copy()
    n_frames = trials_df["RS_stim"].apply(len).to_numpy()
    stim_cols = [c for c in trials_df.columns if c.endswith("_stim")]
    per_frame = [
        c
        for c in stim_cols
        if all(
            hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n
            for v, n in zip(trials_df[c], n_frames)
        )
    ]
    print(f"splitting {len(per_frame)} per-frame columns: {per_frame}")
    half = n_frames // 2
    lo = np.zeros_like(half) if variant == "half1" else n_frames - half
    for c in per_frame:
        trials_df[c] = [
            np.asarray(v)[a : a + h] for v, a, h in zip(trials_df[c], lo, half)
        ]
    start = trials_df["imaging_stim_start"].to_numpy()
    trials_df["imaging_stim_start"] = start + lo
    trials_df["imaging_stim_stop"] = start + lo + half - 1
    trials_df["n_frames_full_window"] = n_frames
    return trials_df


def main(session, variant, out_dir, configs):
    out_dir = Path(out_dir) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = target_config(TARGET, METHOD)
    todo = []
    for model, choose_trials, k_folds in configs:
        base = fit_filename(TARGET, METHOD, model, choose_trials, k_folds)
        out = out_dir / base.replace(".pickle", f"_{variant}.pickle")
        if out.exists():
            print(f"skip (exists): {out.name}")
        else:
            todo.append((model, choose_trials, k_folds, out))
    if not todo:
        return

    t0 = time.time()
    _, _, _, trials_df = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets=FILTER_DATASETS,
        protocol_base=cfg["protocol_base"],
        recording_type="two_photon",
        tread_kwargs={"method": METHOD},
    )
    trials_df = trials_df[~trials_df.recording_name.str.contains("multidepth")]
    print(f"loaded {len(trials_df)} trials in {(time.time() - t0) / 60:.1f} min")
    n_full = trials_df["RS_stim"].apply(len)
    trials_df = split_trials(trials_df, variant)
    n_kept = trials_df["RS_stim"].apply(len)
    print(
        f"frames per trial: full median {n_full.median():.0f} "
        f"[{n_full.min()}-{n_full.max()}], kept median {n_kept.median():.0f}"
    )
    rsof_range = report_rsof_range(trials_df, cfg, label=f"{session} | {variant}")

    for model, choose_trials, k_folds, out in todo:
        t1 = time.time()
        print(
            f"=== {session} | {variant} | {model} choose_trials={choose_trials} k={k_folds}"
        )
        fit_df = fit_gaussian_blob.fit_rs_of_tuning(
            trials_df=trials_df,
            model=model,
            choose_trials=choose_trials,
            trial_sfx="",
            k_folds=k_folds,
            max_rs2motor_diff=cfg["max_rs2motor_diff"],
            trial_average=cfg["trial_average"],
            param_range=cfg["param_range"],
            use_col="dffs",
            **COMMON_PARAMS,
        )
        tmp = out.with_suffix(".pickle.partial")
        fit_df.to_pickle(tmp)
        tmp.replace(out)
        print(f"=== done in {(time.time() - t1) / 60:.1f} min -> {out}")

    record = dict(
        session=session,
        variant=variant,
        target=TARGET,
        method=METHOD,
        configs=[list(c) for c in configs],
        param_range=cfg["param_range"],
        max_rs2motor_diff=cfg["max_rs2motor_diff"],
        common_params=COMMON_PARAMS,
        frames_per_trial_full=n_full.describe().to_dict(),
        frames_per_trial_kept=n_kept.describe().to_dict(),
        rsof_range=rsof_range,
        cottage_analysis_sha=_cottage_analysis_sha(),
    )
    (out_dir / f"provenance_{variant}.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session")
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--configs", nargs="+", default=None)
    args = parser.parse_args()
    configs = (
        [parse_config(c) for c in args.configs] if args.configs else DEFAULT_CONFIGS
    )
    main(args.session, args.variant, args.out_dir, configs)
