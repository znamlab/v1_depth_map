"""Main analysis functions for eye tracking data of the depth project

`hey2_3d-vision_foodres_20220101/Calibrations/cottage_2p/CameraCalibration/notes` lists
the corresponding calibrations for each camera. The notes are as follows:

S202208xx calibrations are for PZAH6.4b & PZAG3.4f
20230406 calibrations are for PZAH8.2h,i,f and PZAH10.2x before 20230601
20230601 calibrations are for PZAH10.2x after 20230601 (right eye cam was changed on
    this day)
20230615 calibrations are for PZAH10.2x after 20230602 (right eye cam was changed on
    this day)
20230626 both lefr and right cams are changed. need recalibration. 0626 10.2d session
    right eye cam was totally out of place and no calibration.
20230824 calibrations were for PZAH10.2x.


Camera were mirrored until 8.2x (excluded).
"""

import cv2
import numpy as np
from pathlib import Path
import flexiznam as flz
import wayla
from cottage_analysis.analysis import spheres
from wayla import eye_model_fitting as emf


CALIBRATION_SESSION = {
    "PZAH6.4b": "20220818",
    "PZAG3.4f": "20220818",
}

IS_MIRRORED = {
    "PZAH6.4b": True,
    "PZAG3.4f": True,
}


def get_data(project, mouse, session, recording, filt_window=5, verbose=True):
    """Get gaze data with behaviour info for a given recording

    Args:
        project (str): Project name
        mouse (str): Mouse name
        session (str): Session name
        recording (str): Recording name
        filt_window (int, optional): Window size for median filter. Defaults to 5.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: Gaze data with behaviour info
    """
    flm_sess = flz.get_flexilims_session(project_id=project)
    dlc_res, gaze_data, dlc_ds = get_gaze(
        project, mouse, session, recording, flm_sess=flm_sess
    )
    trials_df = get_trial_df(mouse, project, session, recording, flm_sess=flm_sess)
    gaze_data = add_behaviour(gaze_data, trials_df)
    gaze_data = cleanup_data(
        gaze_data, dlc_res, filt_window=filt_window, verbose=verbose
    )
    return gaze_data, dlc_res


def get_gaze(
    project,
    mouse,
    session,
    recording,
    verbose=True,
    camera_name="right_eye_camera",
    flm_sess=None,
):
    """Get gaze data for a given recording

    Args:
        project (str): Project name
        mouse (str): Mouse name
        session (str): Session name
        recording (str): Recording name
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        camera_name (str, optional): Camera name. Defaults to 'right_eye_camera'.
        flm_sess (flexilims.Session, optional): Flexilims session. Defaults to None.

    Returns:
        pd.DataFrame: DLC results
        pd.DataFrame: Ellipse fits
        flexiznam.schema.dataset.Dataset: DLC dataset
    """

    flm_sess = flz.get_flexilims_session(project_id=project)
    camera_ds_name = f"{mouse}_{session}_{recording}_{camera_name}"
    camera = flz.Dataset.from_flexilims(name=camera_ds_name, flexilims_session=flm_sess)

    dlc_res, ellipse, dlc_ds = wayla.eye_io.get_data(
        camera,
        flexilims_session=flm_sess,
        likelihood_threshold=0.88,
        rsquare_threshold=0.99,
        maximum_reflection_distance=50,
        error_threshold=None,
        ds_is_cropped=True,
    )

    if verbose:
        print(f"Loaded tracking data for {dlc_res.shape[0]} frames")

    # get calibration data
    if mouse not in CALIBRATION_SESSION:
        raise ValueError(f"No calibration session found for mouse {mouse}")
    calib_session = CALIBRATION_SESSION[mouse]
    if not isinstance(calib_session, str):
        raise NotImplementedError("Only string calib_session is supported")

    # Get camera extrinsics
    processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
    calibration_folder = processed_path / project / "Calibrations"

    calib_data = dict()
    # The folder created by bonsai are camel case
    if camera_name == "right_eye_camera":
        camel_cam_name = "RightEyeCam"
    elif camera_name == "left_eye_camera":
        camel_cam_name = "LeftEyeCam"
    else:
        raise ValueError(f"Unknown camera {camera}")

    folder = calibration_folder / camel_cam_name
    folder = list(folder.glob("*xtrinsics_flat"))[0]  # case is inconsistent
    folder = folder / calib_session / "aruco5_5mm"

    assert folder.exists(), f"Folder {folder} does not exist"
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
    # reshape rvec to match the shape of the output of cv2.solvePnP
    rvec = extrinsics["rvec"].reshape(3, 1)
    if verbose:
        print(f"Loaded camera extrinsics for {camel_cam_name}")

    # apply extrinsics to ellipse
    gaze_vec = wayla.utils.get_gaze_vector(ellipse.phi.values, ellipse.theta.values)
    rotated_gaze_vec = emf.convert_to_world(gaze_vec, rvec=rvec)
    azimuth, elevation = emf.gaze_to_azel(
        rotated_gaze_vec, world_is_mirrored=IS_MIRRORED[mouse]
    )

    ellipse["azimuth"] = np.rad2deg(azimuth)
    ellipse["elevation"] = np.rad2deg(elevation)
    return dlc_res, ellipse, dlc_ds


def get_trial_df(mouse, project, session, recording, flm_sess=None):
    """Get the trial dataframe for a given recording

    Args:
        mouse (str): Mouse name
        project (str): Project name
        session (str): Session name
        recording (str): Recording name
        flm_sess (flexilims.Session, optional): Flexilims session. Defaults to None.

    Returns:
        pd.DataFrame: Trial dataframe
    """
    if flm_sess is None:
        flm_sess = flz.get_flexilims_session(project_id=project)
    # for 2 mice the photodiode protocol is 2, for the rest it's 5
    photodiode_protocol = 2 if mouse in ["PZAH6.4b", "PZAG3.4f"] else 5
    vs_df_all, trials_df = spheres.sync_all_recordings(
        session_name=f"{mouse}_{session}",
        flexilims_session=flm_sess,
        project=project,
        filter_datasets={"anatomical_only": 3},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    n_recordings = len(trials_df.recording_name.unique())
    if n_recordings > 1:
        # filter only for the recording of interest
        trials_df = trials_df[trials_df.recording_name == recording].copy()
    assert recording == trials_df.recording_name.unique()[0]
    return trials_df


def add_behaviour(ellipse, trials_df):
    """Add behaviour information to the ellipse dataframe

    Args:
        ellipse (pd.DataFrame): Ellipse dataframe
        trials_df (pd.DataFrame): Trial dataframe

    Returns:
        pd.DataFrame: Ellipse dataframe with behaviour information, including:
            - is_stim: whether the frame is during the stimulus period
            - trial: trial number
            - depth: depth of the trial
            - RS: Running speed of each frame
            - OF: Optical flow of each frame
    """
    is_stim = np.zeros(len(ellipse), dtype=bool)
    trial = np.zeros(len(ellipse), dtype=float) * np.nan
    depth = np.zeros(len(ellipse), dtype=float) * np.nan
    rs = np.zeros(len(ellipse), dtype=float) * np.nan
    of = np.zeros(len(ellipse), dtype=float)
    harptime = np.zeros(len(ellipse), dtype=float) * np.nan
    for trial_id, row in trials_df.iterrows():
        is_stim[row.imaging_stim_start : row.imaging_stim_stop] = True
        trial[row.imaging_stim_start : row.imaging_stim_stop + 1] = trial_id
        depth[row.imaging_stim_start : row.imaging_stim_stop + 1] = row.depth
        rs[row.imaging_stim_start : row.imaging_stim_stop + 1] = row.RS_stim
        of[row.imaging_stim_start : row.imaging_stim_stop + 1] = row.OF_stim
        # also add rs in the blank period
        rs[row.imaging_blank_pre_start : row.imaging_blank_pre_stop + 1] = (
            row.RS_blank_pre
        )
        # and some known time points
        for side in ["start", "stop"]:
            harptime[row[f"imaging_stim_{side}"]] = row[f"imaging_harptime_stim_{side}"]

    ellipse["is_stim"] = is_stim
    ellipse["trial"] = trial
    ellipse["depth"] = depth
    ellipse["RS"] = rs
    ellipse["OF"] = of
    ellipse["harptime"] = harptime
    # Interpolate missing harptimes
    ellipse["harptime"] = ellipse.harptime.interpolate()
    return ellipse


def cleanup_data(gaze_data, tracking_data, filt_window=5, verbose=True):
    """Clean up gaze data

    Find when the eye is closed and filter out those frames. Fix depth to 2 decimal
    places, interpolate NaN values, and filter with a median filter. Compute
    displacement and velocity.

    Args:
        gaze_data (pd.DataFrame): Gaze data
        tracking_data (pd.DataFrame): Tracking data from deeplabcut
        filt_window (int, optional): Window size for mean filter. Defaults to 5.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: Cleaned gaze data
    """
    # remove the moment when the mouse closes the eye
    eye_open = gaze_data.reflection_dlc_likelihood > 0.8
    gaze_data["valid"] &= eye_open
    tracking_data = tracking_data.copy()
    tracking_data.columns = tracking_data.columns.droplevel(0)
    top = tracking_data["top_eye_lid"]
    bottom = tracking_data["bottom_eye_lid"]
    # calculate the distance between top and bottom
    distance = np.sqrt((top["x"] - bottom["x"]) ** 2 + (top["y"] - bottom["y"]) ** 2)
    gaze_data["eye_is_open"] = distance > np.nanmean(distance) - 3 * np.nanstd(
        distance[gaze_data["valid"]]
    )
    gaze_data["eye_opening"] = distance
    gaze_data["valid"] &= gaze_data["eye_is_open"]
    if verbose:
        print(f"Eye closed in {np.sum(~gaze_data['valid'])}/{len(gaze_data)} frames")
    # round depth
    gaze_data["depth"] = gaze_data["depth"].round(2)
    gaze_data.reset_index(drop=True, inplace=True)

    for col in ["azimuth", "elevation"]:
        # replace NaN with linear interpolation
        gaze_data[f"{col}_interp"] = gaze_data[col].interpolate()
        # filter with a box mean filter
        gaze_data[f"{col}_filt"] = (
            gaze_data[f"{col}_interp"].rolling(window=filt_window, center=True).mean()
        )
        # rolling is actually rolling, NaN the beginning/end that are using data from the
        # opposite end
        gaze_data.loc[: filt_window // 2, f"{col}_filt"] = np.nan
        gaze_data.loc[len(gaze_data) - filt_window // 2 - 1 :, f"{col}_filt"] = np.nan

    dt = np.nanmedian(np.diff(gaze_data["harptime"].values))
    fs = 1 / dt
    print(f"Sampling rate: {fs:.2f} Hz")
    displacement = np.sqrt(
        np.diff(gaze_data["azimuth_filt"]) ** 2
        + np.diff(gaze_data["elevation_filt"]) ** 2
    )
    velocity = displacement / dt
    gaze_data["displacement"] = np.concatenate([[0], displacement])
    gaze_data["velocity"] = np.concatenate([[0], velocity])
    gaze_data["depth_label"] = [f"{d:.2f}m" for d in gaze_data["depth"]]
    return gaze_data
