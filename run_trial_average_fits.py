import os
import warnings
from pathlib import Path
import flexiznam as flz
from cottage_analysis.summary_analysis import get_session_list
from cottage_analysis.pipelines.fit_rsof_trial_average import (
    submit_fitting_array,
    merge_and_concatenate_results,
    PROJECTS,
    SESSIONS_TO_EXCLUDE,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    slurm_folder = Path(os.path.expanduser("~/slurm_logs/trial_average"))
    slurm_folder.mkdir(parents=True, exist_ok=True)

    sessions_by_project = {}

    for project in PROJECTS:
        flexilims_session = flz.get_flexilims_session(project_id=project)
        session_list = get_session_list.get_motor_session_list(
            flexilims_session=flexilims_session,
            exclude_sessions=list(SESSIONS_TO_EXCLUDE.keys()),
        )
        sessions_by_project[project] = session_list

    print(f"Submitting Slurm array job for all sessions...")
    job_id = submit_fitting_array(
        sessions_by_project=sessions_by_project,
        conda_env="v1_depth_map",
        tasks_dir=str(slurm_folder),
    )
    print(f"Submitted Array Job ID: {job_id}")

    if job_id:
        print(f"\nSubmitting merge job depending on: {job_id}...")
        merge_job_id = merge_and_concatenate_results(
            sessions_by_project=sessions_by_project,
            use_slurm=True,
            job_dependency=job_id,
            dependency_type="afterok",
            slurm_folder=str(slurm_folder),
            scripts_name="merge_ta_results",
        )
        print(f"  Submitted Merge Job ID: {merge_job_id}")
    else:
        print("No sessions to fit.")


if __name__ == "__main__":
    main()
