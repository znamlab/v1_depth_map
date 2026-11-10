"""Refit the treadmill 1D depth tuning of one session into a run directory.

Same fits as `rerun_depth_fit.py` (find_depth_neurons, then fit_preferred_depth on all
trials, on odd trials and with 5-fold cross-validation, `max_rs2motor_diff=0.3`), but:

- run with the fixed cottage_analysis seeding: `best_depth` is NaN-aware (trials whose
  frames were all removed by the RS-to-motor filter used to make it the first NaN depth),
  and `fit_preferred_depth` restarts from depths spread over the tested range instead of
  reusing the best-depth seed for every restart. The old seed left most treadmill plateau
  fits flat (preferred depth at a fit bound, R^2 ~ 0);
- the production `neurons_df.pickle` is only read (for ROI alignment), never written. The
  depth columns of the variant (and their mirror, e.g. `_treadmill_plateau` beside
  `_treadmill`) are saved, with `roi`, to
  OUT_DIR/SESSION/treadmill_depth_fit{suffix}.pickle, with provenance in a .json.

Usage:
    python refit_treadmill_depth.py SESSION VARIANT --out-dir DIR
VARIANT is a name from `rerun_depth_fit.VARIANTS` (plateau_single_frames,
model_single_frames).
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

import cottage_analysis
from cottage_analysis.analysis import common_utils, find_depth_neurons
from cottage_analysis.pipelines import pipeline_utils
from v1_depth_map.precompute_data.rerun_depth_fit import (
    DEPTH_COLUMN_KEYS,
    PHOTODIODE_PROTOCOL,
    PROJECT,
    VARIANTS,
)

MAX_RS2MOTOR_DIFF = 0.3


def code_version(module):
    """Git commit (and whether the tree had local changes) of an installed package."""
    root = Path(module.__file__).resolve().parents[1]
    marker = root / "CODE_VERSION.txt"  # written when the code is copied to the run dir
    if marker.exists():
        return marker.read_text().strip()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet"], check=False
        ).returncode
        return commit + (" (local changes)" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(session, variant_name, out_dir):
    var = {v["name"]: v for v in VARIANTS}[variant_name]
    suffix, mirror = var["suffix"], var.get("mirror_suffix")
    out_dir = Path(out_dir) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"treadmill_depth_fit{suffix}.pickle"
    if out.exists():
        raise FileExistsError(f"{out} exists: not overwriting")

    flm = flz.get_flexilims_session(project_id=PROJECT)
    neurons_ds = flz.get_datasets(
        origin_name=session,
        dataset_type="neurons_df",
        flexilims_session=flm,
        allow_multiple=False,
    )
    neurons_df = pd.read_pickle(neurons_ds.path_full)  # read only

    t0 = time.time()
    _, _, _, trials_df = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets={"anatomical_only": 3, "annotated": True},
        protocol_base="SpheresTubeMotor",
        recording_type="two_photon",
        tread_kwargs={"method": var["method"]},
    )
    trials_df = trials_df[~trials_df.recording_name.str.contains("multidepth")]
    print(f"{session}: loaded trials in {time.time() - t0:.0f} s", flush=True)

    ndf, nds = find_depth_neurons.find_depth_neurons(
        trials_df=trials_df,
        neurons_ds=neurons_ds,
        neurons_df=neurons_df.copy(),
        rs_thr=None,
        alpha=0.05,
        special_sfx=suffix,
        max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
    )
    depth_min = np.round(trials_df.depth.min(), 4)
    depth_max = np.ceil(trials_df.depth.max())
    for k_folds, choose_trials in [(1, None), (1, "odd"), (5, None)]:
        t1 = time.time()
        ndf, nds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df,
            neurons_df=ndf,
            neurons_ds=nds,
            depth_min=depth_min,
            depth_max=depth_max,
            k_folds=k_folds,
            choose_trials=choose_trials,
            special_sfx=suffix,
            max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
        )
        print(
            f"{session}: fit k_folds={k_folds} trials={choose_trials} "
            f"in {time.time() - t1:.0f} s",
            flush=True,
        )
    common_utils.add_one_sided_spearman_significance(
        ndf,
        rval_col=f"depth_tuning_test_spearmanr_rval_closedloop{suffix}",
        pval_col=f"depth_tuning_test_spearmanr_pval_closedloop{suffix}",
        out_col=f"is_depth_neuron{suffix}",
    )

    # depth columns of this variant only (endswith is exact: "_treadmill" does not match
    # "_treadmill_model"), plus their mirror twin
    cols = [
        c
        for c in ndf.columns
        if c.endswith(suffix) and any(key in c for key in DEPTH_COLUMN_KEYS)
    ]
    result = ndf[["roi"] + cols].copy()
    if mirror:
        for c in cols:
            result[c[: -len(suffix)] + mirror] = result[c]
    assert result.roi.equals(neurons_df.roi), "ROI order changed"
    result.to_pickle(out)

    old_sel = neurons_df.get(f"is_depth_neuron{suffix}")
    provenance = dict(
        session=session,
        variant=variant_name,
        suffix=suffix,
        mirror_suffix=mirror,
        method=var["method"],
        max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        n_trials=int(len(trials_df)),
        neurons_df=str(neurons_ds.path_full),
        cottage_analysis=code_version(cottage_analysis),
        columns=list(result.columns),
        n_depth_selective_new=int(result[f"is_depth_neuron{suffix}"].sum()),
        n_depth_selective_old=None if old_sel is None else int(old_sel.sum()),
        runtime_s=round(time.time() - t0),
    )
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session")
    parser.add_argument("variant", choices=[v["name"] for v in VARIANTS])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    main(args.session, args.variant, args.out_dir)
