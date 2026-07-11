"""
Run the ridge decoder on the FULL set of conditions (no grid filtering), trained on the
cut treadmill, for all V1 motor sessions across both projects.

This regenerates the non-grid `_motor_cut` parquets:
    ridge_decoder_neurons_motor_cut.parquet
    ridge_decoder_predictions_motor_cut.parquet
    ridge_decoder_trial_averaged_motor_cut.parquet
    ridge_decoder_neuron_subsets_motor_cut.parquet
all of which now include the depth-orthogonal RS x OF target (rsof_product_stim).

Run with:
    /camp/home/blota/.conda/envs/v1_depth_map/bin/python3 run_full_cut.py
"""

import warnings
from pathlib import Path
import os
import flexiznam as flz
from cottage_analysis.summary_analysis import get_session_list
from cottage_analysis.pipelines.ridge_decoder_utils import run_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Session lists ─────────────────────────────────────────────────────────────
PROJECTS = ["ccyp_l5_3d_vision", "colasa_3d-vision_revisions"]
SESSIONS_TO_EXCLUDE = {"PZAG22.1b_S20260220"}

# ── Fixed decoder parameters (match run_grid_subsets.py) ──────────────────────
FILTER_DATASETS = {"anatomical_only": 3, "ast_neuropil": False}
CUT_TREADMILL = True
RUN_NEURON_SUBSETS = True  # needed for the subset-size analyses in NB3 Section 1

# Set to True to submit on Slurm, False to run locally (for testing)
USE_SLURM = True


def main():
    for project in PROJECTS:
        flm_sess = flz.get_flexilims_session(project_id=project)
        motor_sessions = get_session_list.get_motor_session_list(flm_sess)
        motor_sessions = [s for s in motor_sessions if s not in SESSIONS_TO_EXCLUDE]
        print(f"\nProject {project}: {len(motor_sessions)} motor sessions")

        for sess in motor_sessions:
            suffix = "_motor_cut"
            if USE_SLURM:
                slurm_folder = Path(os.path.expanduser(f"~/slurm_logs/{sess}"))
                slurm_folder.mkdir(parents=True, exist_ok=True)
            else:
                slurm_folder = None

            print(f"  Submitting {sess} ...", flush=True)
            try:
                run_session(
                    sess=sess,
                    project=project,
                    filter_datasets=FILTER_DATASETS,
                    cut_treadmill=CUT_TREADMILL,
                    run_neuron_subsets=RUN_NEURON_SUBSETS,
                    is_treadmill=True,
                    grid_keys=None,
                    grid_name=None,
                    use_slurm=USE_SLURM,
                    slurm_folder=slurm_folder,
                    scripts_name=f"ridge_decoder_{sess}{suffix}",
                )
            except Exception as e:
                print(f"  ERROR for {sess}: {e}", flush=True)

    print("\nAll submissions done.")


if __name__ == "__main__":
    main()
