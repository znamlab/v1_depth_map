import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

import flexiznam as flz
from cottage_analysis.analysis import common_utils


def plot_treadmill_protocol(
    trials_df,
    example_trials,
    fs,
    trials_df_no_cut=None,
    max_abs_rs2motor_diff_ratio=0.3,
    plot_exclude_frames=True,
    ax=None,
    save_path=None,
    add_vertical_lines=False,
    figsize=(15 / 2.54, 7 / 2.54),
    xlim=(0, 61),
    ylim=(-0.05, 0.8),
    fontsize_dict=None,
):
    """
    Plots the treadmill protocol including running speed data and stimulus
    presentation periods.

    Args:
        trials_df (pd.DataFrame): DataFrame containing 'RS_stim' and
            'max_abs_rs2motor_diff_ratio_stim'.
        example_trials (list): List of trial indices to plot.
        fs (float): Sampling frequency.
        trials_df_no_cut (pd.DataFrame, optional): DataFrame containing
            'RS_blank_pre', 'RS_stim', 'RS_blank', and 'OF_stim'.
            If None, uses trials_df. Defaults to None.
        max_abs_rs2motor_diff_ratio (float, optional): Threshold for excluding
            frames. Defaults to 0.3.
        plot_exclude_frames (bool, optional): Whether to plot the excluded
            frames. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a
            new figure. Defaults to None.
        save_path (str or Path, optional): Path to save the figure as a PDF.
            Defaults to None.
        add_vertical_lines (bool, optional): Whether to add vertical lines at
            stimulus boundaries. Defaults to False.
        figsize (tuple, optional): Figure size in inches.
            Defaults to (15/2.54, 7/2.54).
        xlim (tuple, optional): X-axis limits, None to leave them automatic.
            Defaults to (0, 61).
        ylim (tuple, optional): Y-axis limits, None to leave them automatic.
            Defaults to (-0.05, 0.8).
        fontsize_dict (dict, optional): Dictionary with font size settings.
            Defaults to None.
    Returns:
        tuple: (fig, ax)
    """
    if trials_df_no_cut is None:
        trials_df_no_cut = trials_df
    if fontsize_dict is None:
        fontsize_dict = {"label": 7, "tick": 5, "legend": 5}
    # Prepare data
    data = None
    of = None
    stim_part = None

    for itrial, trial in enumerate(example_trials):
        trial_series = trials_df_no_cut.loc[trial]
        if itrial == 0:
            data = trial_series.RS_blank_pre[-20:]
            of = np.zeros_like(data) * np.nan
            stim_part = np.zeros(data.shape, dtype=int)

        assert trial_series.RS_stim.shape == trial_series.OF_stim.shape
        data = np.hstack([data, trial_series.RS_stim, trial_series.RS_blank])
        of = np.hstack(
            [
                of,
                trial_series.OF_stim,
                np.zeros_like(trial_series.RS_blank) * np.nan,
            ]
        )
        stim_part = np.hstack(
            [
                stim_part,
                np.ones(trial_series.RS_stim.shape) * trial,
                np.zeros(trial_series.RS_blank.shape),
            ]
        )

    used_data = np.zeros_like(data) * np.nan
    excluded_data = np.zeros_like(data) * np.nan

    for trial in example_trials:
        trial_series = trials_df.loc[trial]
        trial_valid = trial_series.RS_stim
        ok_mask = (
            trial_series.max_abs_rs2motor_diff_ratio_stim < max_abs_rs2motor_diff_ratio
        )
        trial_used = np.where(ok_mask, trial_valid, np.nan)
        trial_excluded = np.where(ok_mask, np.nan, trial_valid)

        trial_indices = np.where(stim_part == trial)[0]
        if len(trial_indices) > 0:
            end_ind = trial_indices[-1]
            excluded_data[end_ind - len(trial_excluded) + 1 : end_ind + 1] = (
                trial_excluded
            )
            used_data[end_ind - len(trial_used) + 1 : end_ind + 1] = trial_used

    # Plotting
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    time_axis = np.arange(len(data)) / fs
    if plot_exclude_frames:
        ax.plot(time_axis, data, color="k", lw=1, clip_on=False, label="All Frames")
        ax.plot(time_axis, excluded_data, color="grey", lw=2, label="Excluded Frames")
    else:
        ax.plot(time_axis, data, color="k", lw=1, clip_on=False)
    ax.plot(
        time_axis,
        used_data,
        color="dodgerblue",
        lw=2,
        label="Analysed Frames",
    )

    has_of = ~np.isnan(of)
    ax.fill_between(
        time_axis,
        0.9,
        1,
        where=has_of,
        color="grey",
        alpha=0.3,
        transform=ax.get_xaxis_transform(),
        zorder=-1,
        label="Sphere Presentation",
    )

    edges = np.where(np.diff(np.concatenate(([False], has_of, [False]))))[0].reshape(
        -1, 2
    )
    if add_vertical_lines:
        for start, end in edges:
            ax.axvline(start / fs, color="black", linestyle="--", alpha=0.5)
            ax.axvline(end / fs, color="black", linestyle="--", alpha=0.5)

    # Scale bar
    x0, y0 = 1, 0.2
    x_len, y_len = 2, 0.1
    ax.plot(
        [x0, x0, x0 + x_len],
        [y0 + y_len, y0, y0],
        color="k",
        lw=2,
        clip_on=False,
        solid_joinstyle="miter",
    )
    ax.text(
        x0 + x_len / 2,
        y0 - (y_len * 0.15),
        f"{int(x_len)} s",
        ha="center",
        va="top",
        fontsize=fontsize_dict["legend"],
        clip_on=False,
    )
    ax.text(
        x0 - (x_len * 0.1),
        y0 + y_len / 2,
        f"{int(y_len*100)} cm/s",
        ha="right",
        va="center",
        rotation=90,
        fontsize=fontsize_dict["legend"],
        clip_on=False,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Random dots for spheres
    for start, end in edges:
        x_start = start / fs + 1
        x_end = end / fs - 1
        n_dots = np.random.randint(4, 8)
        rand_x = np.linspace(x_start, x_end, n_dots) + np.random.uniform(
            -0.5, 0.5, n_dots
        )
        rand_y = np.random.uniform(0.92, 0.98, n_dots)
        ax.scatter(
            rand_x,
            rand_y,
            color="black",
            s=10,
            zorder=0,
            transform=ax.get_xaxis_transform(),
        )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=fontsize_dict["legend"],
        bbox_to_anchor=(0.0, 0.9),
    )

    _save_pdf(fig, save_path)

    return fig, ax


def _save_pdf(fig, save_path):
    """Save a figure as a pdf with editable fonts, if save_path is not None."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    old_pdf_fonttype = mpl.rcParams["pdf.fonttype"]
    old_ps_fonttype = mpl.rcParams["ps.fonttype"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    fig.savefig(save_path, format="pdf", bbox_inches="tight", transparent=True)
    mpl.rcParams["pdf.fonttype"] = old_pdf_fonttype
    mpl.rcParams["ps.fonttype"] = old_ps_fonttype


def plot_treadmill_stim_sampling(
    trials_df,
    ax=None,
    max_abs_rs2motor_diff_ratio=0.3,
    fontsize_dict={"label": 14, "tick": 12, "legend": 12},
    markersize=5,
    trial_markersize=20,
    legend_on=True,
    legend_kwargs=None,
    figsize=(7 / 2.54, 7 / 2.54),
    save_path=None,
):
    """Plot how the treadmill stimulus samples the running speed / optic flow plane.

    Single imaging frames are shown in the background, with the per-trial averages on
    top. Both axes are logarithmic (base 2) and ticked at the motor speeds and expected
    optic flow speeds of the protocol.

    Args:
        trials_df (pd.DataFrame): DataFrame containing 'RS_stim', 'OF_stim',
            'MotorSpeed_stim', 'expected_optic_flow_stim' and, optionally,
            'max_abs_rs2motor_diff_ratio_stim'.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new
            figure. Defaults to None.
        max_abs_rs2motor_diff_ratio (float, optional): Threshold for excluding frames
            where the running speed does not match the motor speed. Set to None to keep
            all frames. Defaults to 0.3.
        fontsize_dict (dict, optional): Dictionary of fontsizes.
        markersize (float, optional): Marker size for single frames. Defaults to 5.
        trial_markersize (float, optional): Marker size for trial averages.
            Defaults to 20.
        legend_on (bool, optional): Whether to add the legend. Defaults to True.
        legend_kwargs (dict, optional): Overrides for the legend keyword arguments.
            Defaults to None.
        figsize (tuple, optional): Figure size in inches, ignored if ax is given.
            Defaults to (7/2.54, 7/2.54).
        save_path (str or Path, optional): Path to save the figure as a PDF.
            Defaults to None.

    Returns:
        tuple: (fig, ax)
    """
    if (max_abs_rs2motor_diff_ratio is not None) and (
        "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns
    ):
        trials_df = common_utils.filter_trials_by_rs2motor(
            trials_df,
            max_rs2motor_diff=max_abs_rs2motor_diff_ratio,
            col2filter=["RS_stim", "OF_stim"],
        )

    # Per-frame running speed (cm/s) and optic flow (degrees/s). Only strictly positive
    # values can be shown on log axes.
    rs_frames = np.hstack(trials_df.RS_stim.values) * 100
    of_frames = np.degrees(np.hstack(trials_df.OF_stim.values))
    ok = (rs_frames > 0) & (of_frames > 0)
    rs_frames, of_frames = rs_frames[ok], of_frames[ok]

    # Per-trial averages, in the same units. Trials without any valid frame are NaN and
    # dropped by the positivity check below.
    def trial_mean(x):
        return np.nanmean(x) if np.any(~np.isnan(x)) else np.nan

    rs_trials = trials_df.RS_stim.map(trial_mean).values * 100
    of_trials = np.degrees(trials_df.OF_stim.map(trial_mean).values)
    ok = (rs_trials > 0) & (of_trials > 0)
    rs_trials, of_trials = rs_trials[ok], of_trials[ok]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    ax.scatter(
        rs_frames,
        of_frames,
        alpha=0.1,
        color="k",
        s=markersize,
        edgecolors="none",
        label="Single frames",
    )
    ax.scatter(
        rs_trials,
        of_trials,
        alpha=0.8,
        color="dodgerblue",
        s=trial_markersize,
        edgecolors="white",
        linewidths=0.5,
        label="Trial averages",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    motor_speeds = np.unique(
        trials_df.MotorSpeed_stim.map(np.nanmedian).dropna().round()
    )
    of_speeds = np.unique(
        trials_df.expected_optic_flow_stim.map(np.nanmedian).dropna().round()
    )
    rs_ticks = motor_speeds[motor_speeds > 0].astype(int)
    of_ticks = of_speeds[of_speeds > 0].astype(int)

    ax.set_xticks(rs_ticks)
    ax.set_xticklabels([f"{x}" for x in rs_ticks], fontsize=fontsize_dict["tick"])
    ax.set_yticks(of_ticks)
    ax.set_yticklabels([f"{x}" for x in of_ticks], fontsize=fontsize_dict["tick"])
    ax.set_xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Optic flow speed (degrees/s)", fontsize=fontsize_dict["label"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(min(rs_ticks), max(rs_ticks))
    ax.spines["left"].set_bounds(min(of_ticks), max(of_ticks))

    if legend_on:
        kwargs = dict(
            loc="upper right",
            bbox_to_anchor=(1, 1.3),
            fontsize=fontsize_dict["legend"],
            frameon=False,
        )
        kwargs.update(legend_kwargs or {})
        ax.legend(**kwargs)

    _save_pdf(fig, save_path)

    return fig, ax
