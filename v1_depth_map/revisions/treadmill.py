"""Analysis of treadmill data"""

import warnings
import numpy as np

STEPS_PER_REV = 200
MICROSTEPPING = 1 / 4
WHEEL_RADIUS = 10.5
circumference = 2 * np.pi * WHEEL_RADIUS


def sps2speed(sps):
    """Convert steps per second to speed in cm/s.

    Args:
        sps (float or list or np.ndarray): Steps per second.

    Returns:
        float or np.ndarray or None: Speed in cm/s. Returns None if input is None.

    """
    if sps is None:
        return None
    if isinstance(sps, list):
        sps = np.array(sps)

    return sps / STEPS_PER_REV * MICROSTEPPING * circumference


def process_imaging_df(imaging_df, trial_duration=2):
    """Process the imaging dataframe to add treadmill information.

    This will take the last `trial_duration` second of each motor step

    The following columns are added:
        - MotorSpeed: Speed of the motor in cm/s.
        - is_trial_end: True if the frame is the end of a trial.
        - is_trial_start: True if the frame is the start of a trial.
        - is_stim: True if the frame is part of a trial.
        - trial_index: Index of the trial.
        - optic_flow: Optic flow in deg/s.

    Args:
        imaging_df (pd.DataFrame): Imaging dataframe.

    Returns:
        pd.DataFrame: Imaging dataframe with treadmill information.
    """

    assert "MotorSps" in imaging_df.columns, "Imaging df must contain MotorSps"

    imaging_df["MotorSpeed"] = np.round(sps2speed(imaging_df.MotorSps))
    # Find trials, defined as last 2 second of motor running
    trial_ends = (imaging_df.MotorSps > 0).astype(int).diff() == -1
    shifted = trial_ends.shift(-1)
    imaging_df["is_trial_end"] = shifted.values.astype(bool)
    trial_starts = (
        imaging_df.loc[imaging_df["is_trial_end"], "imaging_harptime"] - trial_duration
    )
    imaging_df["is_trial_start"] = False
    trial_start_index = imaging_df.imaging_harptime.searchsorted(trial_starts)
    imaging_df.loc[trial_start_index, "is_trial_start"] = True

    starts = imaging_df.query("is_trial_start")
    ends = imaging_df.query("is_trial_end")

    imaging_df["is_stim"] = False
    imaging_df["trial_index"] = -1

    for itrial, (start, end) in enumerate(zip(starts.index, ends.index)):
        imaging_df.loc[start:end, "is_stim"] = True
        imaging_df.loc[start:end, "trial_index"] = itrial

    # Calculate the optic flow
    actual_of = np.rad2deg(imaging_df.MotorSpeed / (imaging_df.depth.values * 100))
    # To get the expected_of we round in the log space

    warnings.filterwarnings("ignore")
    imaging_df["optic_flow"] = 2 ** (np.round(np.log2(actual_of)))
    warnings.filterwarnings("default")

    imaging_df.query("is_stim").optic_flow.value_counts()
    return imaging_df
