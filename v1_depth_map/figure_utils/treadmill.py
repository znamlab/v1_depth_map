import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


def plot_treadmill_protocol(
    trials_df,
    example_trials,
    fs,
    trials_df_no_cut=None,
    max_abs_rs2motor_diff_ratio=0.3,
    ax=None,
    save_path=None,
    add_vertical_lines=False,
    figsize=(15 / 2.54, 7 / 2.54),
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
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a
            new figure. Defaults to None.
        save_path (str or Path, optional): Path to save the figure as a PDF.
            Defaults to None.
        add_vertical_lines (bool, optional): Whether to add vertical lines at
            stimulus boundaries. Defaults to False.
        figsize (tuple, optional): Figure size in inches.
            Defaults to (15/2.54, 7/2.54).
        xlim (tuple, optional): X-axis limits. Defaults to (0, 61).
        ylim (tuple, optional): Y-axis limits. Defaults to (-0.05, 0.8).
    """
    if trials_df_no_cut is None:
        trials_df_no_cut = trials_df

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
    valid_data = np.zeros_like(data) * np.nan

    for trial in example_trials:
        trial_series = trials_df.loc[trial]
        trial_valid = trial_series.RS_stim
        ok_mask = (
            trial_series.max_abs_rs2motor_diff_ratio_stim < max_abs_rs2motor_diff_ratio
        )
        trial_used = np.where(ok_mask, trial_valid, np.nan)

        trial_indices = np.where(stim_part == trial)[0]
        if len(trial_indices) > 0:
            end_ind = trial_indices[-1]
            valid_data[end_ind - len(trial_valid) + 1 : end_ind + 1] = trial_valid
            used_data[end_ind - len(trial_used) + 1 : end_ind + 1] = trial_used

    # Plotting
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    time_axis = np.arange(len(data)) / fs
    ax.plot(time_axis, data, color="k", lw=1, clip_on=False, label="All Frames")
    ax.plot(time_axis, valid_data, color="grey", label="Excluded Frames")
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
        lw=4,
        clip_on=False,
        solid_joinstyle="miter",
    )
    ax.text(
        x0 + x_len / 2,
        y0 - (y_len * 0.15),
        f"{int(x_len)} s",
        ha="center",
        va="top",
        fontsize=10,
        clip_on=False,
    )
    ax.text(
        x0 - (x_len * 0.1),
        y0 + y_len / 2,
        f"{int(y_len*100)} cm/s",
        ha="right",
        va="center",
        rotation=90,
        fontsize=10,
        clip_on=False,
    )

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
            s=25,
            zorder=0,
            transform=ax.get_xaxis_transform(),
        )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.0, 0.9),
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        old_pdf_fonttype = mpl.rcParams["pdf.fonttype"]
        old_ps_fonttype = mpl.rcParams["ps.fonttype"]
        mpl.rcParams["pdf.fonttype"] = 42
        mpl.rcParams["ps.fonttype"] = 42
        fig.savefig(save_path, format="pdf", bbox_inches="tight", transparent=True)
        mpl.rcParams["pdf.fonttype"] = old_pdf_fonttype
        mpl.rcParams["ps.fonttype"] = old_ps_fonttype

    return fig, ax
