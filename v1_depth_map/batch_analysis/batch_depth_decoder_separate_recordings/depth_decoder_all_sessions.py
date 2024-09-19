from cottage_analysis.pipelines import pipeline_utils

project = "hey2_3d-vision_foodres_20220101"
pipeline_filename = "run_depth_decoder_pipeline_separate_recordings.sh"
conflicts = "overwrite"
session_list = [
    "PZAH8.2h_S20230224",
    "PZAH8.2h_S20230303",
    "PZAH8.2h_S20230314",
    
    "PZAH8.2i_S20230203",
    "PZAH8.2i_S20230209",
    "PZAH8.2i_S20230216",

    "PZAH8.2f_S20230214",
    "PZAH8.2f_S20230313",
 
    "PZAH10.2f_S20230615", 
    "PZAH10.2f_S20230822",
    "PZAH10.2f_S20230908",
]

use_slurm = 1
log_fname = "decoder_separate_recordings"


def main(
    project,
    session_list,
    pipeline_filename="run_analysis_pipeline.sh",
    conflicts="overwrite",
    use_slurm=False,
    **kwargs,
):
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
            use_slurm=use_slurm,
            **kwargs,
        )


if __name__ == "__main__":
    main(project, session_list, pipeline_filename, conflicts, 
         use_slurm, 
         log_fname=log_fname)
