from cottage_analysis.pipelines import pipeline_utils
import flexiznam as flz
from v1_depth_analysis.v1_manuscript_2023 import get_session_list

project = "hey2_3d-vision_foodres_20220101"
pipeline_filename = "run_openloop_pipeline.sh"
conflicts = "overwrite"
use_slurm = 1

def main(
    project,
    pipeline_filename="run_openloop_pipeline.sh",
    conflicts="overwrite",
    **kwargs,
):
    flexilims_session = flz.get_flexilims_session(project)
    # session_list = ["PZAH8.2i_S20230203"]
    session_list = get_session_list.get_sessions(
            flexilims_session,
            closedloop_only=False,
            openloop_only=True,
        )
    for session_name in session_list:
        if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5

        pipeline_utils.sbatch_session(
            project=project,
            session_name=session_name,
            pipeline_filename=pipeline_filename,
            conflicts=conflicts,
            photodiode_protocol=photodiode_protocol,
            **kwargs,
        )


if __name__ == "__main__":
    main(project, pipeline_filename, conflicts, 
         use_slurm=use_slurm,)
