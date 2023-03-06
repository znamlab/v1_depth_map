"""
Preprocessing of eye tracking

This selects the sessions and runs ellipse fit jobs to track eye

Ellipse fit must already been generated.
"""
import warnings
from pathlib import Path
import flexiznam as flm
from flexiznam.schema import Dataset
from cottage_analysis import eye_tracking
import v1_depth_analysis as vda
from v1_depth_analysis.config import PROJECT

REDO = False

raw_path = Path(flm.PARAMETERS["data_root"]["raw"])
processed_path = Path(flm.PARAMETERS["data_root"]["processed"])
flm_sess = flm.get_flexilims_session(project_id=PROJECT)

recordings = vda.get_recordings(protocol="SpheresPermTubeReward")
datasets = vda.get_datasets(
    recordings, dataset_type="camera", dataset_name_contains="_eye"
)

process_list = []
for camera_ds in datasets:
    target_folder = Path(processed_path, PROJECT, *camera_ds.genealogy)
    if not len(list(target_folder.glob("*.h5"))):
        warnings.warn(f"No DLC data for {target_folder}")
        continue
    for dlc_file in target_folder.glob("*.h5"):
        ellipse_file = target_folder / dlc_file.name.replace(".h5", "_ellipse_fits.csv")
        if not ellipse_file.exists():
            warnings.warn(f"No ellipse fits for {dlc_file}")
            continue
        target_file = target_folder / dlc_file.name.replace(".h5", "_ellipse_fits.csv")
        if target_file.exists() and not REDO:
            continue

    process = eye_tracking.find_pupil.reproject_pupils(
        camera_ds.full_name, project=PROJECT, target_folder=target_folder
    )
    process_list.append(process)
print(f"Started {len(process_list)} jobs")
