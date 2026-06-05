# %%
import flexiznam as flz
from twop_preprocess import pipeline_utils


# Settings
PROJECT = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(PROJECT)
if False:
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
    SESSION_LIST = pipeline_utils.get_sessions(
        flexilims_session,
        exclude_openloop=False,
        exclude_pure_closedloop=False,
        v1_only=True,
        trialnum_min=10,
        mouse_list=mouse_list,
    )
else:
    SESSION_LIST = ["PZAH10.2d_S20230822", "PZAH10.2f_S20230815", "PZAH10.2f_S20230907"]

PIPELINE_FILENAME = "run_dff_noneuropil.sh"
CONFLICTS = "overwrite"
TAU = 0.7


def main(
    project,
    session_list,
    pipeline_filename=PIPELINE_FILENAME,
    conflicts="skip",
    tau=0.7,
):
    for session_name in session_list:
        pipeline_utils.sbatch_session(
            project=project,
            session_name=session_name,
            pipeline_filename=pipeline_filename,
            conflicts=conflicts,
            tau=tau,
        )


if __name__ == "__main__":
    main(PROJECT, SESSION_LIST, PIPELINE_FILENAME, CONFLICTS, TAU)
