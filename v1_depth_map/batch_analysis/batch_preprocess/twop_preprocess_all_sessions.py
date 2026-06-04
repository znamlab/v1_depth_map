# %%
import flexiznam as flz
import numpy as np
import pandas as pd
from twop_preprocess import pipeline_utils
from cottage_analysis.summary_analysis import get_session_list

# Settings
PROJECT = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(PROJECT)
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
SESSION_LIST = get_session_list.get_sessions(
    flexilims_session,
    exclude_openloop=False,
    exclude_pure_closedloop=False,
    v1_only=True,
    trialnum_min=10,
    mouse_list=mouse_list,
)
""" SESSION_LIST = [ "PZAH6.4b_S20220503",
                "PZAH6.4b_S20220524",
                "PZAG3.4f_S20220520",
                "PZAG3.4f_S20220523"] """
# %%
PIPELINE_FILENAME = "run_dff_noneuropil.sh"
CONFLICTS = "overwrite"
TAU = 0.7


def main(
    project,
    session_list,
    pipeline_filename="run_suite2p_gpu.sh",
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
