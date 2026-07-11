"""Submit run_rf_only for every motor session, one Slurm job per session.

Runs RF-only analysis (``cottage_analysis.pipelines.run_rf_only``) on the same
set of sessions used in ``decoder_treadmill_cut``: all motor sessions across both
projects, minus the manually excluded one. Each session is submitted as its own
Slurm job, which acts as a lightweight driver that fans the RF hyperparameter
grid search out as a nested Slurm array (one array task per reg combination).

Run with:
    /camp/home/blota/.conda/envs/v1_depth_map/bin/python3 submit_rf_only.py
Set DRY_RUN = True to just list the sessions without submitting.
"""

import os
import warnings
from pathlib import Path

import flexiznam as flz

from cottage_analysis.summary_analysis.get_session_list import get_motor_session_list
from cottage_analysis.pipelines.run_rf_only import run_rf_only_session

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Session lists (match run_full_cut.py / decoder_treadmill_cut) ─────────────
PROJECTS = ["ccyp_l5_3d_vision", "colasa_3d-vision_revisions"]
SESSIONS_TO_EXCLUDE = ("PZAG22.1b_S20260220",)
PROTOCOL_BASE = "SpheresTubeMotor"

DRY_RUN = False  # True: list sessions only, do not submit


def main():
    total = 0
    for project in PROJECTS:
        flm = flz.get_flexilims_session(project_id=project)
        sessions = get_motor_session_list(flm, exclude_sessions=SESSIONS_TO_EXCLUDE)
        print(f"\nProject {project}: {len(sessions)} motor sessions")
        for sess in sessions:
            total += 1
            if DRY_RUN:
                print(f"  [dry-run] {sess}")
                continue

            slurm_folder = Path(os.path.expanduser(f"~/slurm_logs/rf_only/{sess}"))
            slurm_folder.mkdir(parents=True, exist_ok=True)

            print(f"  Submitting {sess} ...", flush=True)
            try:
                job_id = run_rf_only_session(
                    project=project,
                    session_name=sess,
                    protocol_base=PROTOCOL_BASE,
                    use_annotated=True,
                    use_slurm=True,
                    slurm_folder=str(slurm_folder),
                    scripts_name=f"rf_only_{sess}",
                )
                print(f"    job: {job_id}")
            except Exception as e:
                print(f"    ERROR for {sess}: {e}", flush=True)

    print(f"\n{'Listed' if DRY_RUN else 'Submitted'} {total} sessions.")


if __name__ == "__main__":
    main()
