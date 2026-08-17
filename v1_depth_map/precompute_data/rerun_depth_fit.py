"""Re-run 1D depth tuning fits for treadmill sessions (model and plateau onset detection).

Variants:
1. _treadmill: method="model" (already computed in production pipeline)
2. _treadmill_plateau: method="plateau"

Note: 1D depth tuning fits (find_depth_neurons.fit_preferred_depth) always reduce each trial
to trial-mean dF/F, so there is no per-frame vs trial-average distinction for depth fits.
Only the onset-detection method ('model' vs 'plateau') distinguishes depth fits.

Appends/updates the computed 1D depth fit columns directly into `neurons_df.pickle` for local processed files.
Only runs locally (asserts local processed data root on /Volumes/BlackPasspo).
"""

import time
import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.analysis import find_depth_neurons

PROJECT = "colasa_3d-vision_revisions"
PHOTODIODE_PROTOCOL = 5
FILTER_DATASETS = dict(annotated=True)

# 2 treadmill depth dataset variants
VARIANTS = [
    {
        "name": "model_single_frames",
        "method": "model",
        "suffix": "_treadmill",
    },
    {
        "name": "plateau_single_frames",
        "method": "plateau",
        "suffix": "_treadmill_plateau",
    },
]


def assert_local_only():
    """Verify processing operates strictly on local disk and NOT nemo."""
    processed_root = str(flz.get_data_root("processed", project=PROJECT))
    print(f"Verified processed root: {processed_root}")
    if "nemo" in processed_root.lower():
        raise RuntimeError(
            f"Refusing to run: processed root is on nemo ({processed_root}). Must be local."
        )
    if not processed_root.startswith("/Volumes/BlackPasspo"):
        raise RuntimeError(
            f"Refusing to run: processed root {processed_root} is not on local drive /Volumes/BlackPasspo."
        )


def update_neurons_df_depth_columns(neurons_df_path, fit_results_df, suffix):
    """Safely append or update depth fit columns directly into neurons_df.pickle without calling merge_fit_dataframes."""
    neurons_df = pd.read_pickle(neurons_df_path)

    # Check ROI alignment
    assert fit_results_df.roi.equals(
        neurons_df.roi
    ), f"ROI mismatch between fit results and {neurons_df_path}"

    # Identify depth columns created for this suffix (strictly ending with suffix)
    depth_cols = [c for c in fit_results_df.columns if c.endswith(suffix)]

    for col in depth_cols:
        neurons_df[col] = fit_results_df[col]

    # Calculate is_depth_neuron{suffix}
    rval_col = f"depth_tuning_test_spearmanr_rval_closedloop{suffix}"
    pval_col = f"depth_tuning_test_spearmanr_pval_closedloop{suffix}"
    if rval_col in neurons_df.columns and pval_col in neurons_df.columns:
        neurons_df[f"is_depth_neuron{suffix}"] = (neurons_df[rval_col] > 0.1) & (
            neurons_df[pval_col] < 0.05
        )

    neurons_df.to_pickle(neurons_df_path)
    print(
        f"Updated {len(depth_cols)} depth columns for suffix '{suffix}' in {neurons_df_path.name}"
    )


def run_depth_fit_for_session(session_name, flexilims_session, skip_existing=False):
    example_mouse, example_session = session_name.split("_")
    print("\n=======================================================")
    print(f"Processing session: {session_name}")
    print("=======================================================")

    neurons_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        allow_multiple=False,
    )
    if neurons_ds is None or not neurons_ds.path_full.exists():
        print(f"Skipping {session_name}: neurons_df dataset not found.")
        return

    neurons_df_path = neurons_ds.path_full

    for var in VARIANTS:
        suffix = var["suffix"]
        method = var["method"]

        pref_depth_col = f"preferred_depth_closedloop_crossval{suffix}"
        if skip_existing:
            curr_df = pd.read_pickle(neurons_df_path)
            if (
                pref_depth_col in curr_df.columns
                and not curr_df[pref_depth_col].isna().all()
            ):
                print(f"--- Skip {var['name']} (exists): {pref_depth_col} ---")
                continue

        print(f"\n--- Fitting variant: {var['name']} (suffix: {suffix}) ---")
        t0 = time.time()

        # Load session once per variant
        _, _, _, trials_df_tm = pipeline_utils.load_session(
            project=PROJECT,
            session_name=session_name,
            photodiode_protocol=PHOTODIODE_PROTOCOL,
            regenerate_frames=False,
            filter_datasets={"anatomical_only": 3, "annotated": True},
            protocol_base="SpheresTubeMotor",
            recording_type="two_photon",
            tread_kwargs={"method": method},
        )
        is_multidepth = trials_df_tm.recording_name.str.contains("multidepth")
        trials_df_tm = trials_df_tm[~is_multidepth]

        neurons_df_curr = pd.read_pickle(neurons_df_path)

        # 1. Run find_depth_neurons (ANOVA & best_depth)
        ndf, nds = find_depth_neurons.find_depth_neurons(
            trials_df=trials_df_tm,
            neurons_ds=neurons_ds,
            neurons_df=neurons_df_curr.copy(),
            rs_thr=None,
            alpha=0.05,
            special_sfx=suffix,
            max_rs2motor_diff=0.3,
        )

        depth_min = np.round(trials_df_tm.depth.min(), 4)
        depth_max = np.ceil(trials_df_tm.depth.max())

        # 2a. Run fit_preferred_depth on all trials (k_folds=1) -> preferred_depth_closedloop{suffix}
        ndf, nds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_tm,
            neurons_df=ndf,
            neurons_ds=nds,
            depth_min=depth_min,
            depth_max=depth_max,
            k_folds=1,
            choose_trials=None,
            special_sfx=suffix,
            max_rs2motor_diff=0.3,
        )

        # 2b. Run fit_preferred_depth on odd trials (k_folds=1) -> preferred_depth_closedloop_crossval{suffix}
        ndf, nds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_tm,
            neurons_df=ndf,
            neurons_ds=nds,
            depth_min=depth_min,
            depth_max=depth_max,
            k_folds=1,
            choose_trials="odd",
            special_sfx=suffix,
            max_rs2motor_diff=0.3,
        )

        # 2c. Run fit_preferred_depth with 5-fold cross-validation -> depth_tuning_test_rsq...
        ndf, nds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df_tm,
            neurons_df=ndf,
            neurons_ds=nds,
            depth_min=depth_min,
            depth_max=depth_max,
            k_folds=5,
            choose_trials=None,
            special_sfx=suffix,
            max_rs2motor_diff=0.3,
        )

        # 3. Direct update to neurons_df.pickle without calling merge_fit_dataframes
        update_neurons_df_depth_columns(neurons_df_path, ndf, suffix)
        print(f"Completed {var['name']} in {(time.time() - t0):.1f} s")


def main():
    assert_local_only()
    flexilims_session = flz.get_flexilims_session(project_id=PROJECT)

    treadmill_sessions = [
        "PZAG16.3b_S20250401",
        "PZAG16.3c_S20250401",
        "PZAG17.3a_S20250402",
        "PZAH17.1e_S20250403",
    ]

    print(
        f"Targeting {len(treadmill_sessions)} treadmill sessions: {treadmill_sessions}"
    )

    for session_name in treadmill_sessions:
        run_depth_fit_for_session(session_name, flexilims_session, skip_existing=True)

    print("\nAll treadmill depth fit variants completed successfully!")


if __name__ == "__main__":
    main()
