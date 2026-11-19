"""Per-trial averages of a treadmill (motor) session's per-frame ``trials_df``.

``reduce_trials_to_average`` reduces a per-frame ``trials_df`` (from
``pipeline_utils.load_session``) to one value per closed-loop trial: the running-masked mean
running speed, optic flow and dF/F of every neuron. It matches the trial-averaging path of
``fit_gaussian_blob.fit_rs_of_tuning(..., trial_average=True)`` and is used by
``precompute_data/fit_revision_treadmill.py --check-range`` to report the RS/OF range of the
data entering the treadmill fits.

The reduction bakes in the running mask (``rs_thr`` / ``max_rs2motor_diff``), so analyses
that need a different frame selection cannot use it.
"""

import numpy as np

# Canonical running mask, matching fit_rs_of_tuning(trial_average=True) as used for the
# treadmill RS/OF fits.
RS_THR = 0.01
MAX_RS2MOTOR_DIFF = 0.3


def reduce_trials_to_average(
    trials_df,
    rs_thr=RS_THR,
    max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
    closed_loop_only=True,
):
    """Reduce a per-frame ``trials_df`` to per-trial averages.

    Replicates the trial-averaging path of
    ``fit_gaussian_blob.fit_rs_of_tuning(..., trial_average=True)``: for each trial, keep the
    frames where the animal is running (``RS_stim`` and ``RS_eye_stim`` above ``rs_thr``,
    finite positive ``OF_stim``, and ``max_abs_rs2motor_diff_ratio_stim`` below
    ``max_rs2motor_diff`` when that column exists), then average over surviving frames.

    Args:
        trials_df (pd.DataFrame): per-frame trials dataframe from ``pipeline_utils.load_session``.
        rs_thr (float): running-speed threshold (m/s).
        max_rs2motor_diff (float): max allowed |RS - motor| ratio per frame; ignored if the
            column is absent.
        closed_loop_only (bool): keep only ``closed_loop == 1`` trials.

    Returns:
        dict or None: ``dict(rs, of, depth_label, dff)`` where ``rs`` (m/s) and ``of`` (rad/s)
            are 1-D arrays of length ``n_trials``, ``depth_label`` is the per-trial depth label,
            and ``dff`` is a ``(n_trials, n_neurons)`` float32 array. ``None`` if no trial has
            any running frame.
    """
    if closed_loop_only and "closed_loop" in trials_df.columns:
        trials_df = trials_df[trials_df.closed_loop == 1]
    has_diff = "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns
    has_depth = "depth_labels" in trials_df.columns

    rs_l, of_l, dff_l, depth_l = [], [], [], []
    for _, tr in trials_df.iterrows():
        rs = np.asarray(tr["RS_stim"], dtype=float)
        rs_eye = np.asarray(tr["RS_eye_stim"], dtype=float)
        of = np.asarray(tr["OF_stim"], dtype=float)
        dff = np.asarray(tr["dff_stim"], dtype=float)
        run = (rs > rs_thr) & (rs_eye > rs_thr) & (~np.isnan(of)) & (of > 0)
        if has_diff:
            run = run & (
                np.asarray(tr["max_abs_rs2motor_diff_ratio_stim"], float)
                < max_rs2motor_diff
            )
        if np.sum(run) == 0:
            continue
        rs_l.append(np.mean(rs[run]))
        of_l.append(np.mean(of[run]))
        dff_l.append(np.mean(dff[run, :], axis=0))
        depth_l.append(tr["depth_labels"][0] if has_depth else np.nan)

    if not rs_l:
        return None
    return dict(
        rs=np.asarray(rs_l),
        of=np.asarray(of_l),
        depth_label=np.asarray(depth_l),
        dff=np.vstack(dff_l).astype(np.float32),
    )
