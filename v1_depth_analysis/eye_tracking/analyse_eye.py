"""
Function to analyse eye tracking data after preprocessing

Used to load calibration data and compute gaze in world coordinates
"""
import cv2
import numpy as np
import flexiznam as flz
from cottage_analysis.eye_tracking.utils import get_gaze_vector
from cottage_analysis.eye_tracking import eye_model_fitting as emf
from cottage_analysis.eye_tracking import analysis as analeyesis
from cottage_analysis.eye_tracking import eye_io

from v1_depth_analysis.config import PROJECT


def get_eye_tracking_data(
    mouse, session, project=PROJECT, camera="right_eye", verbose=True
):
    flm_sess = flz.get_flexilims_session(project_id=project)
    # get the data
    camera_ds, reprojection_ds = get_camera_and_reprojection_datasets(
        mouse, session, flm_sess, camera, verbose
    )
    dlc_res, ellipse, dlc_ds = eye_io.get_data(
        camera_ds,
        flexilims_session=flm_sess,
        likelihood_threshold=0.88,
        rsquare_threshold=0.99,
        error_threshold=3,
    )
    data, sampling = analeyesis.add_behaviour(
        camera_ds, dlc_res, ellipse, speed_threshold=0.01, log_speeds=False
    )
    assert "valid" in data.columns

    reproj_res = np.load(reprojection_ds.path_full)
    for icol, col in enumerate(["phi", "theta", "radius"]):
        data[col] = reproj_res[: len(data), icol]

    if camera != "right_eye":
        raise NotImplementedError("Only right eye implemented")

    azimuth, elevation = gaze_in_world(
        reproj_res,
        camera_name="RightEyeCam",
        extrinsics_session="20220818",
        project_name=project,
        world_is_mirrored=True,
    )
    data["azimuth"] = np.degrees(azimuth[: len(data)])
    data["elevation"] = np.degrees(elevation[: len(data)])
    return data, sampling, dlc_ds, dlc_res, camera_ds


def get_camera_and_reprojection_datasets(
    mouse, session, flexilims_session, camera="right_eye", verbose=True
):
    """Get the dataset for a given camera

    Args:
        mouse (str): Mouse name
        session (str): Session name
        flexilims_session (flexilims.Session): Flexilims session
        camera (str, optional): Camera name. Defaults to "right_eye".
        verbose (bool, optional): Whether to print information. Defaults to True.

    Returns:
        flexiznam.dataset: Camera Dataset object
        flexiznam.dataset: Reprojection Dataset object
    """
    if verbose:
        print(f"Project : {PROJECT}")

    sess = flz.get_entity(
        name=f"{mouse}_{session}",
        datatype="session",
        flexilims_session=flexilims_session,
    )
    if verbose:
        print(f"{sess.name} {sess.id}")
    recs = flz.get_children(
        parent_id=sess.id,
        children_datatype="recording",
        flexilims_session=flexilims_session,
    )
    if verbose:
        print(f"Found {len(recs)} recordings")
    rec = [recs.loc[r] for r in recs.index if "sphere" in r.lower()][0]
    if verbose:
        print(f"Using recording {rec.name} ({rec.id})")
    datasets = flz.get_datasets(
        origin_id=rec.id, flexilims_session=flexilims_session, dataset_type="camera"
    )
    camera_ds = [ds for ds in datasets if camera in ds.dataset_name][0]
    if verbose:
        print(f"Using dataset {camera_ds.full_name}")

    reprojection_ds = flz.Dataset.from_origin(
        origin_id=camera_ds.origin_id,
        dataset_type="eye_reprojection",
        flexilims_session=flexilims_session,
        base_name=f"{camera_ds.dataset_name}_eye_reprojection",
        conflicts="skip",
    )

    return camera_ds, reprojection_ds


def get_extrinsics(camera_name, extrinsics_session="20220818", project_name=PROJECT):
    """Get extrinsics for a camera

    Loads all trials for a given camera and returns the median extrinsics across
    trials.

    Args:
        camera_name (str): Name of the camera
        extrinsics_session (str, optional): Name of the session to use for
            extrinsics. Defaults to "20220818".
        project_name (str, optional): Name of the project. Defaults to PROJECT.

    Returns:
        dict: Dictionary with rvec and tvec
    """
    processed_path = flz.get_data_root("processed", project=project_name)
    calibration_folder = processed_path / project_name / "Calibrations"

    if camera_name not in ["RightEyeCam", "LeftEyeCam"]:
        raise ValueError("camera_name must be RightEyeCam or LeftEyeCam")

    calib_data = dict()
    folder = calibration_folder / camera_name
    folder = list(folder.glob("*xtrinsics_flat"))[0]  # case is inconsistent
    folder = folder / extrinsics_session / "aruco5_5mm"
    assert folder.exists()
    for trial in folder.glob("trial*"):
        fname = str(trial / "camera_extrinsics_flat.yml")
        s = cv2.FileStorage()
        s.open(fname, cv2.FileStorage_READ)
        rvec = s.getNode("rvec").mat()
        tvec = s.getNode("tvec").mat()
        calib_data[trial.name] = dict(rvec=rvec, tvec=tvec)
    # take median across trials
    extrinsics = dict()
    for w in ["rvec", "tvec"]:
        extrinsics[w] = np.median(
            np.vstack([d[w].flatten() for d in calib_data.values()]), axis=0
        )
    return extrinsics


def gaze_in_world(
    reproj_res,
    camera_name,
    extrinsics_session="20220818",
    project_name=PROJECT,
    world_is_mirrored=True,
):
    """Convert gaze in camera coordinates to gaze in world coordinates

    Args:
        reproj_res (numpy.array): Reprojection results, must contain phi and theta
            (radius is not used)
        camera_name (str): Name of the camera
        extrinsics_session (str, optional): Name of the session to use for
            extrinsics. Defaults to "20220818".
        project_name (str, optional): Name of the project. Defaults to v1_depth_analysis.config.PROJECT.
        world_is_mirrored (bool, optional): Whether the world is mirrored. Defaults to True.

    Returns:
        numpy.array: Azimuth in radians
        numpy.array: Elevation in radians
    """

    rvec = get_extrinsics(camera_name, extrinsics_session, project_name)["rvec"]
    gaze_vec = np.vstack([get_gaze_vector(p[0], p[1]) for p in reproj_res])
    world_gaze = emf.convert_to_world(gaze_vec, rvec=rvec)
    azimuth, elevation = emf.gaze_to_azel(
        world_gaze, zero_median=False, worled_is_mirrored=world_is_mirrored
    )

    return azimuth, elevation
