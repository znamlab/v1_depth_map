"""Fit the SIMULATED treadmill responses with the real trial-average plateau config.

`treadmill.simulate_and_fit_session` (which wrote the
`simulated_responses_fit_treadmill_*.parquet` files) fits the circularised-response control
with a configuration that does not match the fits the figures use:

    fit parameter        real (figure)              simulate_and_fit_session
    onset detection      plateau                    whatever sync_all_recordings defaulted
                                                    to when the file was written ("model"
                                                    for the April 2026 artifacts)
    samples              trial average              per frame
    param_range          TREADMILL_PARAM_RANGE      wide DEFAULT_PARAM_RANGE
    k_folds              1 and 5                    1 only (in-sample R^2, no test_rsq)
    niter                10                         5
    max_rs2motor_diff    0.3                        0.5
    seed popt            -                          rsof_popt_closedloop_g2d_treadmill
                                                    (the per-frame family)

So the control and the population it is a control *for* are different fit families, and
their elongation/orientation distributions are not directly comparable. This script
produces a second set of simulated fits that mirror the real ones exactly, seeded from the
same fit family, and writes them to their own parquet. Nothing existing is touched: the
April `simulated_responses_fit_treadmill_2_0.15_circular.parquet` files (and the
`method="model"` override in `figsupp_simulation_control.ipynb` cell 10 that goes with
them) are left alone.

How the config is kept in sync
------------------------------
The fit parameters are not re-declared here. `fit_revision_treadmill.target_config(
"treadmill", "plateau")` is the single source of truth for `trial_average`,
`max_rs2motor_diff` and `param_range`, and `COMMON_PARAMS` for `rs_thr`/`niter`/
`min_sigma`, so if the real fits are re-tuned this follows automatically.

The simulation itself needs no new cottage_analysis code: `pipeline_utils.load_session`
forwards `**tread_kwargs` into `treadmill.sync_all_recordings`, which accepts both `method`
and the `sim_*` arguments. One load therefore yields the plateau-cut trials AND the
simulated dF/F on exactly the trial set the real fits saw, which is what makes a paired
per-ROI real-vs-simulated comparison valid.

Outputs, per session, beside `neurons_df.pickle`:

    simulated_responses_fit_treadmill_trial_average_plateau_2_0.15_circular.parquet
    simulation_fit_current.json     (provenance; NOT param_range_current.json, which
                                     belongs to the real fits and must not be clobbered)

`fake_dff` is deliberately NOT stored - it is already in the April parquet - so the file is
a few KB rather than ~100 MB.

Typical use::

    python fit_revision_simulation.py --dry-run
    python fit_revision_simulation.py --sessions PZAG17.3a_S20250402   # time one first
    python fit_revision_simulation.py                                   # all four

Cost: the session load dominates at a few minutes; the simulation is one convolution per
ROI (seconds); the trial-average g2d fit is ~40 s at k=1 and ~3-4 min at k=5 for ~780 ROIs.
Roughly 10 min per session. Peak memory is the ~50k frame x ~800 ROI dF/F array, held twice
between the simulation and the swap below.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.analysis import fit_gaussian_blob
from cottage_analysis.pipelines import pipeline_utils

from v1_depth_map.precompute_data.fit_revision_treadmill import (
    COMMON_PARAMS,
    FILTER_DATASETS,
    PHOTODIODE_PROTOCOL,
    PROJECT,
    SESSIONS,
    _cottage_analysis_sha,
    assert_site_root,
    target_config,
)

# The real fit family this mirrors. `target_config` resolves FIT_TARGETS for one onset
# method, so `trial_average`, `max_rs2motor_diff` and `param_range` come from there.
TARGET, METHOD = "treadmill", "plateau"

# Which fit the simulation is seeded from: the same trial-average plateau popts whose
# elongation the figure's ELONGATION_CUTOFF is applied to. `simulate_and_fit_session` uses
# the per-frame `rsof_popt_closedloop_g2d_treadmill` instead.
GROUNDTRUTH_COL = "rsof_popt_closedloop_g2d_treadmill_trial_average_plateau"

# Simulation constants, matching rerun_simulation_tdecay2_areanorm.py (and the TDECAY/TRISE
# that figsupp_simulation_control.ipynb passes to the loader).
DECAY_TAU = 2
RISE_TAU = 0.15
MAKE_CIRCULAR = True
KERNEL_NORMALIZATION = "area"

# k=1 gives the popt the ellipse geometry comes from; k=5 gives the cross-validated
# test R^2 (so the simulated population can be thresholded against its own empirical null
# instead of borrowing the real `rsof_neuron_treadmill` flag) and the five fold popts.
K_FOLDS = (1, 5)

# Raw (pre-merge) column names `fit_rs_of_tuning` writes, and what they are called in the
# parquet. `protocol_sfx` is "closedloop" and `trial_sfx` is "" (with `choose_trials=None`,
# `choose_trials_subset` resets it), hence these names.
K1_COLUMNS = {
    "rsof_popt_closedloop_g2d": "popt_simulated",
    "rsof_rsq_closedloop_g2d": "rsq_simulated",
    "rsof_minSigma_closedloop_g2d": "min_sigma",
}
K5_COLUMNS = {
    "rsof_test_rsq_closedloop_g2d": "test_rsq_simulated",
    "rsof_train_popt_closedloop_g2d": "train_popt_simulated",
}

PROVENANCE_FILENAME = "simulation_fit_current.json"


def output_filename():
    """Parquet name, extending the loader's pattern with a longer `which` tag."""
    is_circ = "_circular" if MAKE_CIRCULAR else "_elliptical"
    return (
        f"simulated_responses_fit_treadmill_trial_average_{METHOD}"
        f"_{DECAY_TAU}_{RISE_TAU}{is_circ}.parquet"
    )


def load_neurons_df(session_name):
    """The session's `neurons_df` plus its dataset, without touching the recordings."""
    flexilims_session = flz.get_flexilims_session(PROJECT)
    neurons_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        allow_multiple=False,
    )
    if neurons_ds is None:
        raise ValueError(f"Neurons dataset not found for session {session_name}")
    return neurons_ds, pd.read_pickle(neurons_ds.path_full)


def seed_popt_list(neurons_df):
    """Ground-truth popts to simulate from, `None` where there is no usable fit.

    Same guard as `simulate_and_fit_session`. `GROUNDTRUTH_COL` is only populated where the
    real trial-average fit succeeded, so expect more `None`s than the per-frame seed gives;
    `simulate_calcium_responses` skips those ROIs and leaves their trace NaN.
    """
    if GROUNDTRUTH_COL not in neurons_df.columns:
        raise KeyError(
            f"{GROUNDTRUTH_COL} not in neurons_df. Run fit_revision_treadmill.py "
            "(--only treadmill --method plateau) and then --merge first."
        )
    popt_list = [
        None
        if (isinstance(popt, float) or np.isnan(popt).any())
        else np.asarray(popt).copy()
        for popt in neurons_df[GROUNDTRUTH_COL].values
    ]
    n_seed = sum(popt is not None for popt in popt_list)
    print(f"### seed popts: {n_seed}/{len(popt_list)} ROIs from {GROUNDTRUTH_COL}")
    return popt_list, n_seed


def write_provenance(session_dir, cfg, n_seed, n_rois):
    """Record what produced the parquet - the fit records only `min_sigma`.

    Deliberately NOT `param_range_current.json`: that filename belongs to
    `fit_revision_treadmill.py`'s real fits and is rewritten per run, so sharing it would
    destroy the real fits' provenance.
    """
    record = {
        "script": Path(__file__).name,
        "target": TARGET,
        "method": METHOD,
        "groundtruth_col": GROUNDTRUTH_COL,
        "n_seed_popts": n_seed,
        "n_rois": n_rois,
        "decay_tau": DECAY_TAU,
        "rise_tau": RISE_TAU,
        "make_circular": MAKE_CIRCULAR,
        "kernel_normalization": KERNEL_NORMALIZATION,
        "k_folds": list(K_FOLDS),
        "param_range": cfg["param_range"],
        "trial_average": cfg["trial_average"],
        "max_rs2motor_diff": cfg["max_rs2motor_diff"],
        "min_sigma": COMMON_PARAMS["min_sigma"],
        "rs_thr": COMMON_PARAMS["rs_thr"],
        "niter": COMMON_PARAMS["niter"],
        "cottage_analysis_sha": _cottage_analysis_sha(),
    }
    path = Path(session_dir) / PROVENANCE_FILENAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"### wrote provenance record: {path}")
    return path


def _rename_fit_columns(fit_df, mapping, k_folds):
    """Pull the columns of interest out of a fit result, renamed for the parquet."""
    missing = [c for c in mapping if c not in fit_df.columns]
    if missing:
        raise KeyError(
            f"k={k_folds} fit result is missing {missing}; got {sorted(fit_df.columns)}"
        )
    out = fit_df[list(mapping)].rename(columns=mapping)
    out["roi"] = (
        fit_df["roi"].values if "roi" in fit_df.columns else np.asarray(out.index)
    )
    return out


def fit_one_session(session_name, skip_existing=True):
    """Simulate and fit one session. Returns the results dataframe, or None if skipped."""
    cfg = target_config(TARGET, METHOD)
    neurons_ds, neurons_df = load_neurons_df(session_name)
    out_path = neurons_ds.path_full.with_name(output_filename())
    if skip_existing and out_path.exists():
        print(f"--- skip (exists): {out_path.name}")
        return None

    popt_list, n_seed = seed_popt_list(neurons_df)

    # One load, shared by both fits. `tread_kwargs` reaches `sync_all_recordings`
    # (pipeline_utils.load_session forwards `**(tread_kwargs or {})`), so the same call
    # both cuts trials with `method` and simulates the responses.
    t0 = time.time()
    _, _, _, trials_df = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session_name,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets=FILTER_DATASETS,
        protocol_base=cfg["protocol_base"],
        recording_type="two_photon",
        tread_kwargs={
            "method": METHOD,
            "sim_popt_list": popt_list,
            "sim_tau_decay": DECAY_TAU,
            "sim_tau_rise": RISE_TAU,
            "sim_make_circular": MAKE_CIRCULAR,
            "sim_kernel_normalization": KERNEL_NORMALIZATION,
        },
    )
    print(f"### loaded and simulated in {(time.time() - t0) / 60:.1f} min")

    # Same exclusion as fit_revision_treadmill.fit_session_target.
    is_multidepth = trials_df.recording_name.str.contains("multidepth")
    trials_df = trials_df[~is_multidepth]

    if "fake_dff_stim" not in trials_df.columns:
        raise RuntimeError(
            "trials_df has no `fake_dff_stim` - the sim_* arguments did not reach "
            "sync_all_recordings. Check pipeline_utils.load_session's tread_kwargs "
            "forwarding and that protocol_base is SpheresTubeMotor."
        )
    for real, fake in zip(trials_df.dff_stim, trials_df.fake_dff_stim):
        assert real.shape == fake.shape, (
            f"simulated trace shape {fake.shape} != real {real.shape}; the simulation "
            "was not sliced on the same trials as the data"
        )
    # `pop` rather than a plain assignment: both arrays are ~50k frames x ~800 ROIs, and
    # nothing downstream needs the real dF/F.
    trials_df["dff_stim"] = trials_df.pop("fake_dff_stim")
    print(f"### fitting {len(trials_df)} trials of simulated dF/F")

    results = None
    for k_folds in K_FOLDS:
        t0 = time.time()
        fit_df = fit_gaussian_blob.fit_rs_of_tuning(
            trials_df=trials_df,
            model="gaussian_2d",
            choose_trials=None,
            trial_sfx="",
            k_folds=k_folds,
            max_rs2motor_diff=cfg["max_rs2motor_diff"],
            trial_average=cfg["trial_average"],
            param_range=cfg["param_range"],
            **COMMON_PARAMS,
        )
        print(f"### k={k_folds} fit in {(time.time() - t0) / 60:.1f} min")
        part = _rename_fit_columns(
            fit_df, K1_COLUMNS if k_folds == 1 else K5_COLUMNS, k_folds
        )
        results = (
            part if results is None else results.merge(part, on="roi", how="outer")
        )

    results["popt_groundtruth"] = pd.Series(popt_list, dtype=object).values[
        results.roi.values
    ]
    # Column order: roi first, then the ground truth, then the fit outputs.
    ordered = ["roi", "popt_groundtruth"] + [
        c for c in results.columns if c not in ("roi", "popt_groundtruth")
    ]
    results = results[ordered]

    tmp = out_path.with_suffix(".parquet.partial")
    results.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    print(f"### wrote {out_path} ({out_path.stat().st_size / 1e3:.1f} kB)")
    write_provenance(out_path.parent, cfg, n_seed, len(results))
    report_recovered_shape(results, cfg)
    return results


def report_recovered_shape(results, cfg):
    """Sanity check: the ground truth is circular, so this is the elongation null."""
    min_sigma = float(
        pd.to_numeric(results.min_sigma, errors="coerce").dropna().iloc[0]
    )
    popts = results.popt_simulated
    ok = popts.apply(lambda p: isinstance(p, (list, np.ndarray)) and len(p) >= 6)
    with np.errstate(over="ignore", invalid="ignore"):
        elongation = popts[ok].apply(
            lambda p: fit_gaussian_blob.get_semimajor_length(p, min_sigma=min_sigma)
            / fit_gaussian_blob.get_semiminor_length(p, min_sigma=min_sigma)
        )
    n_nan = int((~ok).sum())
    print(
        f"### recovered fits: {int(ok.sum())}/{len(results)} usable ({n_nan} without a "
        f"popt), min_sigma={min_sigma}"
    )
    if len(elongation):
        print(
            f"### elongation of the circular null: median {elongation.median():.2f}, "
            f"{(elongation > 1.4).mean() * 100:.0f}% above 1.4, "
            f"{(elongation > 1.5).mean() * 100:.0f}% above 1.5"
        )
    print(
        f"### test R^2: median {pd.to_numeric(results.test_rsq_simulated).median():.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sessions", nargs="+", default=SESSIONS)
    parser.add_argument(
        "--redo", action="store_true", help="refit even if the parquet exists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be written, without loading any recording",
    )
    parser.add_argument("--site", choices=sorted(("local", "nemo")), default="local")
    args = parser.parse_args()

    assert_site_root(args.site)
    cfg = target_config(TARGET, METHOD)
    print(
        f"mirroring target={TARGET!r} method={METHOD!r}: trial_average="
        f"{cfg['trial_average']}, max_rs2motor_diff={cfg['max_rs2motor_diff']}, "
        f"niter={COMMON_PARAMS['niter']}, min_sigma={COMMON_PARAMS['min_sigma']}, "
        f"k_folds={list(K_FOLDS)}\nparam_range={cfg['param_range']}"
    )

    if args.dry_run:
        for session_name in args.sessions:
            neurons_ds, neurons_df = load_neurons_df(session_name)
            out_path = neurons_ds.path_full.with_name(output_filename())
            has_seed = GROUNDTRUTH_COL in neurons_df.columns
            n_seed = (
                int(
                    neurons_df[GROUNDTRUTH_COL]
                    .apply(lambda p: not (isinstance(p, float) or np.isnan(p).any()))
                    .sum()
                )
                if has_seed
                else 0
            )
            state = "exists" if out_path.exists() else "would write"
            print(
                f"{session_name}: {state} {out_path.name} | "
                f"seed column {'present' if has_seed else 'MISSING'} "
                f"({n_seed}/{len(neurons_df)} ROIs)"
            )
        return

    t_start = time.time()
    for session_name in args.sessions:
        print(f"\n=== {session_name}")
        fit_one_session(session_name, skip_existing=not args.redo)
    print(f"\ntotal: {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
