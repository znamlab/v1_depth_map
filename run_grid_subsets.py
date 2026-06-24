"""
Run ridge decoder subset analysis on two specific grids of conditions.

Grid 1: High OF conditions (OF in 64, 256, 1024; RS in 3.8125, 7, 15, 30, 60)
Grid 2: Low OF conditions (OF in 1, 4, 16, 64; various RS values)

Run with:
    /camp/home/blota/.conda/envs/v1_depth_map/bin/python3 run_grid_subsets.py
"""

import warnings
from pathlib import Path
import os
import flexiznam as flz
from cottage_analysis.summary_analysis import get_session_list
from cottage_analysis.pipelines.ridge_decoder_utils import run_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Grid definitions ──────────────────────────────────────────────────────────
GRID1_KEYS = [
    (60.0, 1024.0),
    (7.0, 256.0),
    (3.8125, 1024.0),
    (15.0, 64.0),
    (3.8125, 64.0),
    (60.0, 256.0),
    (30.0, 256.0),
    (3.8125, 256.0),
    (30.0, 64.0),
    (15.0, 256.0),
    (7.0, 1024.0),
    (30.0, 1024.0),
]

GRID2_KEYS = [
    (60.0, 1.0),
    (3.8125, 4.0),
    (60.0, 4.0),
    (30.0, 4.0),
    (3.8125, 16.0),
    (7.0, 16.0),
    (15.0, 16.0),
    (30.0, 16.0),
    (60.0, 16.0),
    (3.8125, 64.0),
    (7.0, 64.0),
    (60.0, 64.0),
]

GRIDS = {
    "grid1": GRID1_KEYS,
    "grid2": GRID2_KEYS,
}

# ── Session lists ─────────────────────────────────────────────────────────────
PROJECTS = ["ccyp_l5_3d_vision", "colasa_3d-vision_revisions"]
SESSIONS_TO_EXCLUDE = {"PZAG22.1b_S20260220"}

# ── Fixed decoder parameters (match existing pipeline) ────────────────────────
FILTER_DATASETS = {"anatomical_only": 3, "ast_neuropil": False}
CUT_TREADMILL = True
RUN_NEURON_SUBSETS = True

# Set to True to submit on Slurm, False to run locally (for testing)
USE_SLURM = True


def main():
    for project in PROJECTS:
        flm_sess = flz.get_flexilims_session(project_id=project)
        motor_sessions = get_session_list.get_motor_session_list(flm_sess)
        motor_sessions = [s for s in motor_sessions if s not in SESSIONS_TO_EXCLUDE]
        print(f"\nProject {project}: {len(motor_sessions)} motor sessions")

        for grid_name, grid_keys in GRIDS.items():
            print(f"\n  Grid: {grid_name} ({len(grid_keys)} conditions)")
            for sess in motor_sessions:
                suffix = f"_motor_cut_{grid_name}"
                if USE_SLURM:
                    slurm_folder = Path(os.path.expanduser(f"~/slurm_logs/{sess}"))
                    slurm_folder.mkdir(parents=True, exist_ok=True)
                else:
                    slurm_folder = None

                print(f"    Submitting {sess} ...", flush=True)
                try:
                    run_session(
                        sess=sess,
                        project=project,
                        filter_datasets=FILTER_DATASETS,
                        cut_treadmill=CUT_TREADMILL,
                        run_neuron_subsets=RUN_NEURON_SUBSETS,
                        is_treadmill=True,
                        grid_keys=grid_keys,
                        grid_name=grid_name,
                        use_slurm=USE_SLURM,
                        slurm_folder=slurm_folder,
                        scripts_name=f"ridge_decoder_{sess}{suffix}",
                    )
                except Exception as e:
                    print(f"    ERROR for {sess}: {e}", flush=True)

    print("\nAll submissions done.")


if __name__ == "__main__":
    main()
