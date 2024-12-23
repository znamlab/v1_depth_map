from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.summary_analysis import get_session_list
import flexiznam as flz

project = "hey2_3d-vision_foodres_20220101"
pipeline_filename = "run_depth_SFTF.sh"
conflicts = "overwrite"
flexilims_session = flz.get_flexilims_session(project)
mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)
mouse_list = mouse_list[
    mouse_list.name.isin(
        [
            "PZAH6.4b",
            "PZAG3.4f",
            "PZAH8.2h",
            "PZAH8.2i",
            "PZAH8.2f",
            "PZAH10.2d",
            "PZAH10.2f",
        ]
    )
]
# session_list = get_session_list.get_sessions(
#     flexilims_session,
#     exclude_openloop=False,
#     exclude_pure_closedloop=False,
#     v1_only=True,
#     trialnum_min=10,
#     mouse_list=mouse_list,
# )
session_list = [
    # "PZAH10.2d_S20230920",
    # "PZAH10.2d_S20230922",
    "PZAH10.2f_S20230817",
    # "PZAH10.2f_S20230912",
    # "PZAH10.2f_S20230914",
    # "PZAH10.2f_S20230924",
    
]

def main(
    project,
    session_list,
    pipeline_filename="run_depth_SFTF.sh",
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
    main(
        project,
        session_list,
        pipeline_filename,
        conflicts,
    )
