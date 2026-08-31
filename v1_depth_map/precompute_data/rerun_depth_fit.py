"""Re-run 1D depth tuning fits for treadmill sessions (model and plateau onset detection).

Variants:
1. _treadmill: method="plateau" (the production default since cottage_analysis c4ea1cd),
   mirrored to _treadmill_plateau as a self-documenting twin
2. _treadmill_model: method="model"

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
from cottage_analysis.analysis import find_depth_neurons, common_utils

PROJECT = "colasa_3d-vision_revisions"
PHOTODIODE_PROTOCOL = 5
FILTER_DATASETS = dict(annotated=True)

# Substrings marking a column as 1D depth-tuning output rather than RS/OF. Used to keep
# the write-back below to the columns this script actually produced.
DEPTH_COLUMN_KEYS = (
    "preferred_depth",
    "depth_tuning",
    "best_depth",
    "is_depth_neuron",
    "depth_neuron_anova",
)

# 2 treadmill depth dataset variants.
#
# The onset method is always explicit in the output name. `_treadmill` IS the plateau
# family -- it is read at ~240 hardcoded sites across the figures and keeps its name -- and
# `_treadmill_plateau` is maintained alongside it as a self-documenting twin (which is what
# figure_rsof_integration.ipynb reads as `depth_sfx`). `mirror_suffix` keeps the twin in
# lockstep so it can never drift from the family it names.
# See revisions/migrate_treadmill_columns.py for the full convention.
VARIANTS = [
    {
        "name": "plateau_single_frames",
        "method": "plateau",
        "suffix": "_treadmill",
        "mirror_suffix": "_treadmill_plateau",
    },
    {
        "name": "model_single_frames",
        "method": "model",
        "suffix": "_treadmill_model",
        "mirror_suffix": None,
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


def update_neurons_df_depth_columns(
    neurons_df_path, fit_results_df, suffix, mirror_suffix=None
):
    """Safely append or update depth fit columns directly into neurons_df.pickle without calling merge_fit_dataframes.

    If `mirror_suffix` is given, every written column is also copied under that suffix, so
    a self-documenting twin family (e.g. `_treadmill_plateau` beside `_treadmill`) stays
    byte-identical to the family it names instead of silently going stale.
    """
    neurons_df = pd.read_pickle(neurons_df_path)

    # Check ROI alignment
    assert fit_results_df.roi.equals(
        neurons_df.roi
    ), f"ROI mismatch between fit results and {neurons_df_path}"

    # Identify depth columns created for this suffix (strictly ending with suffix).
    # `endswith` is exact, so suffix="_treadmill" does not pick up "_treadmill_model" or
    # "_treadmill_plateau" columns that are already in the frame. It does, however, match
    # the *RS/OF* columns of the same family (fit_results_df starts as a copy of the whole
    # neurons_df), so restrict to depth columns -- otherwise this writes ~84 RS/OF columns
    # straight back onto themselves, which is wasteful and only harmless by luck.
    depth_cols = [
        c
        for c in fit_results_df.columns
        if c.endswith(suffix) and any(key in c for key in DEPTH_COLUMN_KEYS)
    ]

    for col in depth_cols:
        neurons_df[col] = fit_results_df[col]

    # Calculate is_depth_neuron{suffix}
    rval_col = f"depth_tuning_test_spearmanr_rval_closedloop{suffix}"
    pval_col = f"depth_tuning_test_spearmanr_pval_closedloop{suffix}"
    if rval_col in neurons_df.columns and pval_col in neurons_df.columns:
        common_utils.add_one_sided_spearman_significance(
            neurons_df,
            rval_col=rval_col,
            pval_col=pval_col,
            out_col=f"is_depth_neuron{suffix}",
        )

    n_mirrored = 0
    if mirror_suffix:
        written = depth_cols + [f"is_depth_neuron{suffix}"]
        for col in written:
            if col not in neurons_df.columns:
                continue
            neurons_df[col[: -len(suffix)] + mirror_suffix] = neurons_df[col]
            n_mirrored += 1

    neurons_df.to_pickle(neurons_df_path)
    print(
        f"Updated {len(depth_cols)} depth columns for suffix '{suffix}' in {neurons_df_path.name}"
        + (f" (+{n_mirrored} mirrored to '{mirror_suffix}')" if mirror_suffix else "")
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
        # A variant counts as done only if the cross-validated Spearman statistics are
        # there too, not just the preferred depth. They are what
        # `add_one_sided_spearman_significance` needs, so without them
        # `is_depth_neuron{suffix}` comes out all-False and silently empties every
        # downstream depth-cell selection -- which is exactly how the treadmill plateau
        # fits looked "present but complete" while carrying no depth neurons at all.
        stat_cols = [
            f"depth_tuning_test_spearmanr_rval_closedloop{suffix}",
            f"depth_tuning_test_spearmanr_pval_closedloop{suffix}",
        ]
        if skip_existing:
            curr_df = pd.read_pickle(neurons_df_path)
            have = [
                c
                for c in [pref_depth_col] + stat_cols
                if c in curr_df.columns and not curr_df[c].isna().all()
            ]
            if len(have) == 3:
                print(f"--- Skip {var['name']} (complete): {pref_depth_col} ---")
                continue
            if have:
                missing = set([pref_depth_col] + stat_cols) - set(have)
                print(
                    f"--- Refitting {var['name']}: present but INCOMPLETE, "
                    f"empty/missing {sorted(missing)} ---"
                )

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
        update_neurons_df_depth_columns(
            neurons_df_path, ndf, suffix, mirror_suffix=var.get("mirror_suffix")
        )
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
