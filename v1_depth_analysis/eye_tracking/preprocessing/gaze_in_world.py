"""Transform gaze from camera to world coordinates"""

import warnings
import cv2
from pathlib import Path
import flexiznam as flm
import numpy as np
from cottage_analysis.eye_tracking import eye_model_fitting as emf
import v1_depth_analysis as vda
from v1_depth_analysis.config import PROJECT

REDO = False
IS_FLIPPED = True

raw_path = Path(flm.PARAMETERS["data_root"]["raw"])
processed_path = Path(flm.PARAMETERS["data_root"]["processed"])
flm_sess = flm.get_flexilims_session(project_id=PROJECT)

recordings = vda.get_recordings(protocol="SpheresPermTubeReward")
datasets = vda.get_datasets(
    recordings, dataset_type="camera", dataset_name_contains="_eye"
)

# get calibration data
calibration_folder = processed_path / PROJECT / "Calibrations"
calib_data = dict()
for cam_name in ["RightEyeCam", "LeftEyeCam"]:
    calib_data[cam_name.lower()] = dict()
    folder = calibration_folder / cam_name
    folder = list(folder.glob("*xtrinsics_flat"))[0]  # case is inconsistent
    folder = folder / "20220818" / "aruco5_5mm"
    assert folder.exists()
    for trial in folder.glob("trial*"):
        fname = str(trial / "camera_extrinsics_flat.yml")
        s = cv2.FileStorage()
        s.open(fname, cv2.FileStorage_READ)
        rvec = s.getNode("rvec").mat()
        tvec = s.getNode("tvec").mat()
        calib_data[cam_name.lower()][trial.name] = dict(rvec=rvec, tvec=tvec)
# take median across trials
extrinsics = dict()
for cam, trials in calib_data.items():
    extrinsics[cam] = dict()
    for w in ["rvec", "tvec"]:
        extrinsics[cam][w] = np.median(
            np.vstack([d[w].flatten() for d in trials.values()]), axis=0
        )
extrinsics[cam]

process_list = []
for camera_ds in datasets:
    target_folder = Path(processed_path, PROJECT, *camera_ds.genealogy)
    # get gaze datasets
    gaze_file = target_folder / f"{camera_ds.dataset_name}_eye_rotation_by_frame.npy"
    if not gaze_file.exists():
        warnings.warn(f"No gaze data for {target_folder}")
        continue
    print(f"Doing {camera_ds.full_name}")
    gaze_camera = np.load(gaze_file)
    # get the camera we need for this acq and build tform matrix
    extrin = extrinsics[camera_ds.dataset_name.replace("_", "")[:-3]]
    tform = np.zeros((4, 4))
    tform[3, 3] = 1
    rmat, jac = cv2.Rodrigues(extrin["rvec"])
    tform[:3, :3] = rmat
    tform[:3, 3] = extrin["tvec"]
    gaze_vec = np.vstack([emf.get_gaze_vector(p[0], p[1]) for p in gaze_camera])
    flipped_gaze = np.array(gaze_vec, copy=True)
    if IS_FLIPPED:
        flipped_gaze[:, 0] *= -1  # because camera are flipped
    gaze_world = (rmat @ flipped_gaze.T).T

    azimuth = np.arctan2(gaze_world[:, 1], gaze_world[:, 0])
    elevation = np.arctan2(gaze_world[:, 2], np.sum(gaze_world[:, :2] ** 2, axis=1))

    target = target_folder / f"{camera_ds.dataset_name}_gaze_in_world.npz"
    np.savez(target, gaze=gaze_world, azimuth=azimuth, elevation=elevation)
    print(f"     saved {target.name}.")
