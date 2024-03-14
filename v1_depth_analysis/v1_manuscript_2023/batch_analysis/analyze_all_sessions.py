from cottage_analysis.pipelines import pipeline_utils

project = "hey2_3d-vision_foodres_20220101"
pipeline_filename = "run_analysis_pipeline.sh"
conflicts = "overwrite"
session_list = [
    # "PZAH10.2f_S20230817",
    # "PZAH10.2f_S20230727",
    # "PZAH10.2d_S20230704",
    "PZAH8.2f_S20230313",
]
run_rf = 1
run_rsof_fit = 0
run_plot = 1

def main(
    project,
    session_list,
    pipeline_filename="run_analysis_pipeline.sh",
    conflicts="overwrite",
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
            **kwargs,
        )


if __name__ == "__main__":
    main(project, session_list, pipeline_filename, conflicts, run_rf=run_rf, run_rsof_fit=run_rsof_fit, run_plot=run_plot)
