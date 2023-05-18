"""
Preprocessing of eye tracking

This selects the sessions and runs DLC jobs to track eye
"""
import os
from pathlib import Path
import flexiznam as flz
import numpy as np
import yaml
import pandas as pd
from flexiznam.schema import Dataset
from cottage_analysis import eye_tracking
from cottage_analysis.eye_tracking import run_dlc
import v1_depth_analysis as vda
from v1_depth_analysis.config import PROJECT, DLC_MODEL


"""
NOTES:

Normal usage is to call this function twice:

- initial tracking:
    CROP = False
- refined tracking:
    CROP = True
"""


if __name__ == "__main__":
    REDO = False  # erase previous tracking and redo
    CROP = False  # track using crop file info

    raw_path = Path(flz.PARAMETERS["data_root"]["raw"])
    processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
    flm_sess = flz.get_flexilims_session(project_id=PROJECT)

    recordings = vda.get_recordings(protocol="SpheresPermTubeReward", flm_sess=flm_sess)
    datasets = vda.get_datasets(
        recordings,
        dataset_type="camera",
        dataset_name_contains="_eye",
        flm_sess=flm_sess,
    )

    which = "cropped" if CROP else "uncropped"
    for camera_ds in datasets:
        run_dlc(camera_ds, flm_sess, dlc_model=DLC_MODEL, crop=CROP, redo=REDO)
