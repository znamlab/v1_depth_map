import functools

print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pickle
from tqdm import tqdm
import scipy

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    common_utils,
    fit_gaussian_blob,
    size_control,
)
from cottage_analysis.plotting import basic_vis_plots, plotting_utils
from cottage_analysis.pipelines import pipeline_utils


def plot_speed_tuning(
    fig,
    neurons_df,
    trials_df,
    roi,
    is_closed_loop,
    nbins=20,
    which_speed="RS",
    speed_min=0.01,
    speed_max=1.5,
    speed_thr=0.01,
    smoothing_sd=1,
    markersize=5,
    linewidth=1,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    legend_on=False,
):
    """Plot a neuron's speed tuning to either running speed or optic flow speed.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        nbins (int, optional): number of bins to bin the tuning curve. Defaults to 20.
        which_speed (str, optional): 'RS': running speed; 'OF': optic flow speed. Defaults to 'RS'.
        speed_min (float, optional): min RS speed for the bins (m/s). Defaults to 0.01.
        speed_max (float, optional): max RS speed for the bins (m/s). Defaults to 1.5.
        speed_thr (float, optional): thresholding RS for logging (m/s). Defaults to 0.01.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")

    if which_speed == "RS":
        speed_tuning = np.zeros(((len(depth_list) + 1), nbins))
        speed_ci = np.zeros(((len(depth_list) + 1), nbins))
        bins = (
            np.linspace(start=speed_min, stop=speed_max, num=nbins + 1, endpoint=True)
            * 100
        )

    elif which_speed == "OF":
        speed_tuning = np.zeros(((len(depth_list)), nbins))
        speed_ci = np.zeros(((len(depth_list)), nbins))
    bin_centers = np.zeros(((len(depth_list)), nbins))

    # Find all speed and dff of this ROI for a specific depth
    for idepth, depth in enumerate(depth_list):
        all_speed = grouped_trials.get_group(depth)[f"{which_speed}_stim"].values
        speed_arr = np.array([j for i in all_speed for j in i])
        all_dff = grouped_trials.get_group(depth)["dff_stim"].values
        dff_arr = np.array([j for i in all_dff for j in i[:, roi]])

        if which_speed == "OF":
            speed_arr = np.degrees(speed_arr)  # rad --> degrees
        if which_speed == "RS":
            speed_arr = speed_arr * 100  # m/s --> cm/s
        # threshold speed
        dff_arr = dff_arr[speed_arr > speed_thr]
        speed_arr = speed_arr[speed_arr > speed_thr]

        if which_speed == "OF":
            bins = np.geomspace(
                start=np.nanmin(speed_arr),
                stop=np.nanmax(speed_arr),
                num=nbins + 1,
                endpoint=True,
            )
        bin_centers[idepth] = (bins[:-1] + bins[1:]) / 2

        # calculate speed tuning
        bin_means, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="mean",
            bins=bins,
        )

        bin_stds, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="std",
            bins=bins,
        )

        bin_counts, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="count",
            bins=bins,
        )

        tuning = plotting_utils.get_tuning_function(
            bin_means, bin_counts, smoothing_sd=smoothing_sd
        )

        ci_range = 0.95
        z = scipy.stats.norm.ppf(1 - ((1 - ci_range) / 2))
        ci = z * bin_stds / np.sqrt(bin_counts)
        ci[np.isnan(ci)] = 0

        speed_tuning[idepth] = tuning
        speed_ci[idepth] = ci

    # Find tuning for blank period for RS
    if which_speed == "RS":
        all_speed = trials_df[f"{which_speed}_blank"].values
        speed_arr = np.array([j for i in all_speed for j in i]) * 100
        all_dff = trials_df["dff_blank"].values
        dff_arr = np.array([j for i in all_dff for j in i[:, roi]])

        # threshold speed
        dff_arr = dff_arr[speed_arr > speed_thr]
        speed_arr = speed_arr[speed_arr > speed_thr]

        # calculate speed tuning
        bin_means, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="mean",
            bins=bins,
        )

        bin_stds, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="std",
            bins=bins,
        )

        bin_counts, _, _ = scipy.stats.binned_statistic(
            x=speed_arr,
            values=dff_arr,
            statistic="count",
            bins=bins,
        )

        tuning = plotting_utils.get_tuning_function(bin_means, bin_counts)

        ci = z * bin_stds / np.sqrt(bin_counts)
        ci[np.isnan(ci)] = 0

        speed_tuning[-1] = tuning
        speed_ci[-1] = ci

    # Plotting
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    for idepth, depth in enumerate(depth_list):
        linecolor = basic_vis_plots.get_depth_color(
            depth, depth_list, cmap=cm.cool.reversed()
        )
        ax.plot(
            bin_centers[idepth, :],
            speed_tuning[idepth, :],
            color=linecolor,
            label=f"{int(depth_list[idepth] * 100)} cm",
        )
        ax.errorbar(
            x=bin_centers[idepth, :],
            y=speed_tuning[idepth, :],
            yerr=speed_ci[idepth, :],
            fmt="o",
            color=linecolor,
            ls="none",
            markersize=markersize,
            linewidth=linewidth,
        )

        if which_speed == "OF":
            ax.set_xscale("log")
            ax.set_xlabel(
                "Optic flow speed \n(degrees/s)", fontsize=fontsize_dict["label"]
            )

    # Plot tuning to gray period
    if which_speed == "RS":
        ax.plot(
            bin_centers[-1, :],
            speed_tuning[-1, :],
            color="gray",
            label=f"blank",
        )
        ax.errorbar(
            x=bin_centers[-1, :],
            y=speed_tuning[-1, :],
            yerr=speed_ci[-1, :],
            fmt="o",
            color="gray",
            ls="none",
            markersize=markersize,
            linewidth=linewidth,
        )
        ax.set_xlabel("Running speed \n(cm/s)", fontsize=fontsize_dict["label"])
        ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])

    if legend_on:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            fontsize=fontsize_dict["legend"],
            frameon=False,
        )
    plotting_utils.despine()


def plot_RS_OF_matrix(
    fig,
    trials_df,
    roi,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
    xlabel="Running speed (cm/s)",
    ylabel="Optical flow speed \n(degrees/s)",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    extended_matrix = np.zeros((log_range["rs_bin_num"], log_range["of_bin_num"]))
    rs_bins = (
        np.logspace(
            log_range["rs_bin_log_min"],
            log_range["rs_bin_log_max"],
            num=log_range["rs_bin_num"],
            base=log_range["log_base"],
        )
        # / 100
    )
    rs_bins = np.insert(rs_bins, 0, 0)

    of_bins = np.logspace(
        log_range["of_bin_log_min"],
        log_range["of_bin_log_max"],
        num=log_range["of_bin_num"],
        base=log_range["log_base"],
    )
    of_bins = np.insert(of_bins, 0, 0)

    rs_arr = np.array([j for i in trials_df.RS_stim.values for j in i]) * 100
    of_arr = np.degrees([j for i in trials_df.OF_stim.values for j in i])
    dff_arr = np.vstack(trials_df.dff_stim.values)[:, roi]

    bin_means, rs_edges, of_egdes, _ = scipy.stats.binned_statistic_2d(
        x=rs_arr, y=of_arr, values=dff_arr, statistic="mean", bins=[rs_bins, of_bins]
    )

    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    im = ax.imshow(
        bin_means[1:, 1:].T,
        origin="lower",
        aspect="equal",
        # cmap=generate_cmap(cmap_name="WhRd"),
        cmap="Reds",
        vmin=0,
        vmax=np.nanmax(bin_means[1:, 1:]),
    )
    (
        ticks_select1,
        ticks_select2,
        bin_edges1,
        bin_edges2,
    ) = basic_vis_plots.get_RS_OF_heatmap_axis_ticks(
        log_range=log_range, fontsize_dict=fontsize_dict
    )
    plt.xticks(
        ticks_select1,
        bin_edges1,
        rotation=45,
        ha="center",
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(ticks_select2, bin_edges2, fontsize=fontsize_dict["tick"])

    ax2 = fig.add_axes([plot_x + plot_width * 0.75, plot_y, cbar_width, plot_height])
    fig.colorbar(im, cax=ax2, label="\u0394F/F")

    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])

    extended_matrix = bin_means
    return extended_matrix
