## DELETE THIS FILE

import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
import scipy
import seaborn as sns
import flexiznam as flz
from scipy.stats import pearsonr
from cottage_analysis.analysis import (
    spheres,
    find_depth_neurons,
    common_utils,
    size_control,
    fit_gaussian_blob,
)
from cottage_analysis.plotting import basic_vis_plots, plotting_utils
from cottage_analysis.pipelines import pipeline_utils
from v1_depth_map.figure_utils import rf
from v1_depth_map.figure_utils.roi_location import find_roi_centers
from v1_depth_map.figure_utils import common_utils as plt_common_utils


def plot_raster_all_depths(
    trials_df,
    roi,
    is_closed_loop,
    corridor_length=6,
    blank_length=0,
    nbins=60,
    vmax=1,
    plot=True,
    cbar_width=0.05,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    position=(0, 0, 1, 1),
):
    """Raster plot for neuronal activity for each depth

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        max_distance (int, optional): max distance for each trial in meters. Defaults to 6.
        nbins (int, optional): number of bins to bin the activity. Defaults to 60.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
        vmax (int, optional): vmax to plot the heatmap. Defaults to 1.
        plot (bool, optional): whether to plot or not. Defaults to True.
        cbar_width (float, optional): width of the colorbar. Defaults to 0.05.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.

    """
    # choose the trials with closed or open loop to visualize
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped = trials_df.groupby(by="depth")
    trial_number = len(trials_df) // len(depth_list)

    # bin dff according to distance travelled for each trial
    dffs_binned = np.zeros((len(depth_list), trial_number, nbins))
    min_distance = -blank_length
    max_distance = corridor_length + blank_length
    bins = np.linspace(
        start=min_distance, stop=max_distance, num=nbins + 1, endpoint=True
    )
    for idepth, depth in enumerate(depth_list):
        for itrial in np.arange(trial_number):
            dff = np.concatenate(
                (
                    grouped.get_group(depth).dff_blank_pre.values[itrial][:, roi],
                    grouped.get_group(depth).dff_stim.values[itrial][:, roi],
                    grouped.get_group(depth).dff_blank.values[itrial][:, roi],
                )
            )
            pos_arr = np.concatenate(
                (
                    grouped.get_group(depth).mouse_z_harp_blank_pre.values[itrial],
                    grouped.get_group(depth).mouse_z_harp_stim.values[itrial],
                    grouped.get_group(depth).mouse_z_harp_blank.values[itrial],
                )
            )
            pos_arr -= grouped.get_group(depth).mouse_z_harp_stim.values[itrial][0]
            bin_means, _, _ = scipy.stats.binned_statistic(
                x=pos_arr,
                values=dff,
                statistic="mean",
                bins=bins,
            )
            dffs_binned[idepth, itrial, :] = bin_means

    # colormap
    WhRdcmap = basic_vis_plots.generate_cmap(cmap_name="WhRd")

    # plot all depths as one heatmap
    if plot:
        plot_x, plot_y, plot_width, plot_height = position
        plot_prop = 0.9
        each_plot_width = (plot_width - cbar_width) / len(depth_list)
        ax = plt.gcf().add_axes([plot_x, plot_y, plot_width, plot_height])
        im = ax.imshow(
            np.swapaxes(dffs_binned, 0, 1).reshape(-1, nbins * len(depth_list)),
            aspect="auto",
            cmap=WhRdcmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )
        # Plot vertical lines to separate different depths
        ndepths = len(depth_list)
        for i in range(ndepths - 1):
            ax.axvline((i + 1) * nbins, color="k", linewidth=0.5, linestyle="dotted")
        # Change y ticks to trial number
        ax.set_ylabel("Trial number", fontsize=fontsize_dict["label"], labelpad=-5)
        ax.set_yticks([-0.5, dffs_binned.shape[1] - 0.5])
        ax.set_yticklabels([1, dffs_binned.shape[1]])
        ax.tick_params(axis="y", labelsize=fontsize_dict["tick"])
        # Change xticks positions to the middle of current ticks and show depth at the tick position
        blank_prop = blank_length / (corridor_length + blank_length * 2)
        xticks = np.linspace(nbins / 2, nbins * (ndepths - 1 / 2), ndepths)
        ax.set_xticks(xticks)
        ax.set_xticklabels((np.array(depth_list) * 100).astype("int"))
        ax.set_xlabel("Virtual depth (cm)", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="x", labelsize=fontsize_dict["tick"], rotation=0)

        # # for aligning with the scalebar
        # ax.vlines(blank_prop*nbins, 0, dffs_binned.shape[1], color="k", linestyle="--", linewidth=0.5)
        # ax.vlines(nbins-blank_prop*nbins, 0, dffs_binned.shape[1], color="k", linestyle="--", linewidth=0.5)

        ax2 = plt.gcf().add_axes(
            [
                plot_x + plot_width + 0.01,
                plot_y,
                cbar_width * 0.8,
                plot_height / 3,
            ]
        )
        # set colorbar
        cbar = plt.colorbar(im, cax=ax2, label="\u0394F/F")
        ax2.tick_params(labelsize=fontsize_dict["tick"])
        cbar.set_ticks([0, vmax])
        ax2.set_ylabel("\u0394F/F", fontsize=fontsize_dict["legend"])

    return dffs_binned, ax


def plot_depth_tuning_curve(
    neurons_df,
    trials_df,
    roi,
    param="depth",
    use_col="depth_tuning_popt_closedloop",
    min_sigma=0.5,
    folds=None,
    rs_thr=None,
    rs_thr_max=None,
    still_only=False,
    still_time=0,
    frame_rate=15,
    plot_fit=True,
    plot_smooth=False,
    linewidth=3,
    linecolor="k",
    markersize=5,
    markeredgecolor="k",
    closed_loop=1,
    label=None,
    ylim=None,
    ylim_precision_base=1,
    ylim_precision=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """
    Plot depth tuning curve for one neuron.

    Args:
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        rs_thr (float, optional): Threshold to cut off non-running frames. Defaults to 0.2. (m/s)
        still_only (bool, optional): Whether to use only frames when the mouse stay still for x frames. Defaults to False.
        still_time (int, optional): Number of seconds to use when the mouse stay still. Defaults to 0.
        frame_rate (int, optional): Imaging frame rate. Defaults to 15.
        plot_fit (bool, optional): Whether to plot fitted tuning curve or not. Defaults to True.
        linewidth (int, optional): linewidth. Defaults to 3.
        linecolor (str, optional): linecolor of true data. Defaults to "k".
        fit_linecolor (str, optional): linecolor of fitted curve. Defaults to "r".

    """

    # Load average activity and confidence interval for this roi
    trials_df = trials_df[trials_df.closed_loop == closed_loop]
    if param == "depth":
        param_list = np.array(find_depth_neurons.find_depth_list(trials_df))
    elif param == "size":
        trials_df = size_control.get_physical_size(
            trials_df, use_cols=["size", "depth"], k=1
        )
        param_list = np.sort(trials_df["physical_size"].unique())
    log_param_list = np.log(param_list)
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
        trials_df=trials_df,
        rs_thr=rs_thr,
        rs_thr_max=rs_thr_max,
        still_only=still_only,
        still_time=still_time,
        frame_rate=frame_rate,
        closed_loop=closed_loop,
        param=param,
    )[:, :, roi]
    CI_low, CI_high = common_utils.get_bootstrap_ci(mean_dff_arr)
    mean_arr = np.nanmean(mean_dff_arr, axis=1)
    ax = plt.gca()
    ax.errorbar(
        log_param_list,
        mean_arr,
        yerr=(mean_arr - CI_low, CI_high - mean_arr),
        fmt=".",
        color=linecolor,
        markeredgecolor=markeredgecolor,
        markeredgewidth=0.3,
        markerfacecolor=linecolor,
        ls="none",
        fillstyle="full",
        linewidth=linewidth,
        markersize=markersize,
    )

    if plot_smooth:
        # calculate a tuning curve using gaussian smoothing over depths
        xs = np.linspace(log_param_list[0], log_param_list[-1], num=100)
        sd = 0.75
        ys = np.zeros((len(xs)))
        for i, x in enumerate(xs):
            weights = np.exp(-((log_param_list - x) ** 2) / (2 * sd**2))
            ys[i] = np.sum(weights * mean_arr) / np.sum(weights)
        plt.plot(
            xs,
            ys,
            color=linecolor,
            label=label,
            linewidth=linewidth,
        )
    # Load gaussian fit params for this roi
    if plot_fit:
        x = np.geomspace(param_list[0], param_list[-1], num=100)
        if folds is not None:
            for fold in np.arange(folds):
                [a, x0, log_sigma, b] = neurons_df.loc[roi, use_col][fold]
                gaussian_arr = fit_gaussian_blob.gaussian_1d(
                    np.log(x), a, x0, log_sigma, b, min_sigma
                )
                plt.plot(
                    np.log(x),
                    gaussian_arr,
                    color=linecolor,
                    linewidth=linewidth,
                    label=label,
                )
        else:
            [a, x0, log_sigma, b] = neurons_df.loc[roi, use_col]
            gaussian_arr = fit_gaussian_blob.gaussian_1d(
                np.log(x), a, x0, log_sigma, b, min_sigma
            )
            plt.plot(
                np.log(x),
                gaussian_arr,
                color=linecolor,
                linewidth=linewidth,
                label=label,
            )
    if ylim is None:
        ylim = [
            plt.gca().get_ylim()[0],
            plt_common_utils.ceil(np.max(CI_high), ylim_precision_base, ylim_precision),
        ]
        plt.ylim(ylim)
        plt.yticks(
            [
                0,
                plt_common_utils.ceil(
                    np.max(CI_high), ylim_precision_base, ylim_precision
                ),
            ],
            fontsize=fontsize_dict["tick"],
        )
    else:
        plt.ylim([ylim[0], ylim[1]])
        plt.yticks([np.round(ylim[0], 1), ylim[1]], fontsize=fontsize_dict["tick"])

    if param == "depth":
        plt.xticks(
            log_param_list,
            (np.round(np.array(param_list) * 100)).astype("int"),
        )
        plt.xlabel(f"Virtual depth (cm)", fontsize=fontsize_dict["label"])
    elif param == "size":
        plt.xticks(
            log_param_list,
            np.round(np.array(param_list) * 0.87 / 10 * 20, 1),
        )
        plt.xlabel(f"Virtual radius (cm)", fontsize=fontsize_dict["label"])
    sns.despine(ax=plt.gca(), offset=3, trim=True)
    plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"], labelpad=-5)
    plt.xticks(
        rotation=45,
    )
    plt.gca().tick_params(axis="both", labelsize=fontsize_dict["tick"])


def get_PSTH(
    roi,
    psth=[],
    depth_list=[],
    is_closed_loop=1,
    trials_df=None,
    use_col="dff",
    rs_thr_min=None,  # m/s
    rs_thr_max=None,  # m/s
    still_only=False,
    still_time=1,  # s
    max_distance=6,  # m
    min_distance=0,
    nbins=20,
    bins=[],  # if bins are provided, nbins is ignored
    frame_rate=15,
    compute_ci=True,
):
    # confidence interval z calculation
    ci_range = 0.95

    if len(bins) == 0:
        if len(depth_list) > 0:
            all_ci = np.zeros((2, len(depth_list) + 1, nbins))
        bins = np.linspace(
            start=min_distance, stop=max_distance, num=nbins + 1, endpoint=True
        )
        bin_centers = (bins[1:] + bins[:-1]) / 2
    else:
        nbins = len(bins) - 1
        if len(depth_list) > 0:
            all_ci = np.zeros((2, len(depth_list) + 1, len(bins)-1))
        bin_centers = (bins[1:] + bins[:-1]) / 2
    if len(psth) == 0:
        # choose the trials with closed or open loop to visualize
        trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

        depth_list = find_depth_neurons.find_depth_list(trials_df)
        grouped = trials_df.groupby(by="depth")
        # trial_number = len(trials_df) // len(depth_list)

        # bin dff according to distance travelled for each trial
        all_means = np.zeros((len(depth_list) + 1, nbins))
        all_ci = np.zeros((2, len(depth_list) + 1, nbins))

        all_trial_numbers = []
        for idepth, depth in enumerate(depth_list):
            all_trial_numbers.append(len(grouped.get_group(depth)))
        trial_number = np.min(all_trial_numbers)

        for idepth, depth in enumerate(depth_list):
            all_dff = []
            for itrial in np.arange(trial_number):
                # concatenate dff_blank_pre, dff, and dff_blank
                if use_col == "dff":
                    dff = np.concatenate(
                        (
                            grouped.get_group(depth)[f"{use_col}_blank_pre"].values[
                                itrial
                            ][:, roi],
                            grouped.get_group(depth)[f"{use_col}_stim"].values[itrial][
                                :, roi
                            ],
                            grouped.get_group(depth)[f"{use_col}_blank"].values[itrial][
                                :, roi
                            ],
                        )
                    )
                else:
                    dff = np.concatenate(
                        (
                            grouped.get_group(depth)[f"{use_col}_blank_pre"].values[
                                itrial
                            ],
                            grouped.get_group(depth)[f"{use_col}_stim"].values[itrial],
                            grouped.get_group(depth)[f"{use_col}_blank"].values[itrial],
                        )
                    )
                rs_arr = np.concatenate(
                    (
                        grouped.get_group(depth).RS_blank_pre.values[itrial],
                        grouped.get_group(depth).RS_stim.values[itrial],
                        grouped.get_group(depth).RS_blank.values[itrial],
                    )
                )
                pos_arr = np.concatenate(
                    (
                        grouped.get_group(depth).mouse_z_harp_blank_pre.values[itrial],
                        grouped.get_group(depth).mouse_z_harp_stim.values[itrial],
                        grouped.get_group(depth).mouse_z_harp_blank.values[itrial],
                    )
                )
                pos_arr -= grouped.get_group(depth).mouse_z_harp_stim.values[itrial][0]

                take_idx = apply_rs_threshold(
                    rs_arr, rs_thr_min, rs_thr_max, still_only, still_time, frame_rate
                )

                dff = dff[take_idx]
                # bin dff according to distance travelled
                dff, _, _ = scipy.stats.binned_statistic(
                    x=pos_arr,
                    values=dff,
                    statistic="mean",
                    bins=bins,
                )
                all_dff.append(dff)
            all_means[idepth, :] = np.nanmean(all_dff, axis=0)
            if compute_ci:
                all_ci[0, idepth, :], all_ci[1, idepth, :] = (
                    common_utils.get_bootstrap_ci(
                        np.array(all_dff).T, sig_level=1 - ci_range
                    )
                )
    else:
        all_dff = psth
        all_means = np.nanmean(all_dff, axis=0)
        for idepth, depth in enumerate(depth_list):
            if compute_ci:
                all_ci[0, idepth, :], all_ci[1, idepth, :] = (
                    common_utils.get_bootstrap_ci(
                        np.array(all_dff[:, idepth, :]).T, sig_level=1 - ci_range
                    )
                )

    return all_means, all_ci, bin_centers


def apply_rs_threshold(
    rs_arr, rs_thr_min, rs_thr_max, still_only, still_time, frame_rate
):
    # threshold running speed according to rs_thr
    if not still_only:  # take running frames
        if (rs_thr_min is None) and (rs_thr_max is None):  # take all frames
            take_idx = np.arange(len(rs_arr))
        else:
            if rs_thr_max is None:  # take frames with running speed > rs_thr
                take_idx = rs_arr > rs_thr_min
            elif rs_thr_min is None:  # take frames with running speed < rs_thr
                take_idx = rs_arr < rs_thr_max
            else:  # take frames with running speed between rs_thr_min and rs_thr_max
                take_idx = (rs_arr > rs_thr_min) & (rs_arr < rs_thr_max)
    else:  # take still frames
        if rs_thr_max is None:  # use not running data but didn't set rs_thr
            print(
                "ERROR: calculating under not_running condition without rs_thr_max to determine max speed"
            )
        else:  # take frames with running speed < rs_thr for x seconds
            take_idx = common_utils.find_thresh_sequence(
                array=rs_arr,
                threshold_max=rs_thr_max,
                length=int(still_time * frame_rate),
                shift=int(still_time * frame_rate),
            )
    return take_idx


def plot_PSTH(
    roi,
    is_closed_loop,
    trials_df=None,
    psth=[],
    depth_list=[],
    use_col="dff",
    corridor_length=6,
    blank_length=0,
    nbins=20,
    bins=[],  # if bins are provided, nbins is ignored
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    frame_rate=15,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    linewidth=3,
    legend_on=False,
    legend_loc="lower right",
    legend_bbox_to_anchor=(1.4, -0.6),
    show_ci=True,
    ylim=(None, None),
):
    """PSTH of a neuron for each depth and blank period.

    Args:
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        max_distance (int, optional): max distance for each trial in meters. Defaults to 6.
        nbins (int, optional): number of bins to bin the activity. Defaults to 20.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
    """
    max_distance = corridor_length + blank_length
    min_distance = -blank_length
    if trials_df is not None:
        depth_list = find_depth_neurons.find_depth_list(trials_df)

    all_means, all_ci, bin_centers = get_PSTH(
        trials_df=trials_df,
        psth=psth,
        depth_list=depth_list,
        roi=roi,
        is_closed_loop=is_closed_loop,
        use_col=use_col,
        rs_thr_min=rs_thr_min,
        rs_thr_max=rs_thr_max,
        still_only=still_only,
        still_time=still_time,
        min_distance=min_distance,
        max_distance=max_distance,
        nbins=nbins,
        bins=bins,
        frame_rate=frame_rate,
    )

    if use_col == "RS":
        all_means = all_means * 100  # convert m/s to cm/s
    for idepth, depth in enumerate(depth_list):
        linecolor = plotting_utils.get_color(
            depth_list[idepth], 
            value_min=np.min(depth_list),
            value_max=np.max(depth_list),
            log=True,
            cmap=cm.cool.reversed()
        )
        plt.plot(
            bin_centers,
            all_means[idepth, :],
            color=linecolor,
            label=f"{(np.round(depth_list[idepth] * 100)).astype('int')} cm",
            linewidth=linewidth,
        )
        if show_ci:
            plt.fill_between(
                bin_centers,
                y1=all_ci[0, idepth, :],
                y2=all_ci[1, idepth, :],
                color=linecolor,
                alpha=0.3,
                edgecolor=None,
                rasterized=False,
            )

    plt.xlabel("Corridor position (m)", fontsize=fontsize_dict["label"])
    plt.ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
    plt.xticks(
        [0, corridor_length],
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(fontsize=fontsize_dict["tick"])
    if (ylim[0] is None) and (ylim[1] is None):
        ylim = plt.gca().get_ylim()
        ylim = [ylim[0], plt_common_utils.ceil(ylim[1], 1)]
    elif ylim[0] is not None:
        if ylim[1] is None:
            ylim = (ylim[0], plt_common_utils.ceil(plt.gca().get_ylim()[1], 1))
            plt.ylim(ylim)
        else:
            ylim = ylim
        plt.ylim(ylim)
    elif (ylim[1] is not None) and (ylim[0] is None):
        ylim = (plt.gca().get_ylim()[0], ylim[1])
        plt.ylim(ylim)
    plt.yticks([ylim[0], ylim[1]], fontsize=fontsize_dict["tick"])
    plt.plot([0, 0], 
             ylim, 
             "k", 
             linestyle="dotted", 
             linewidth=0.5, 
             label="_nolegend_")
    plt.plot(
        [corridor_length, corridor_length],
        ylim,
        "k",
        linestyle="dotted",
        linewidth=0.5,
        label="_nolegend_",
    )

    if legend_on:
        plt.legend(
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            fontsize=fontsize_dict["legend"],
            frameon=False,
            handlelength=1,
        )
    plotting_utils.despine()


def get_psth_crossval_all_sessions(
    flexilims_session,
    session_list,
    nbins=10,
    closed_loop=1,
    use_cols=[
        "preferred_depth_closedloop_crossval",
        "depth_tuning_test_rsq_closedloop",
    ],
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    verbose=1,
    corridor_length=6,
    blank_length=0,
    overwrite=False,
):
    """Calculate the PSTH for all sessions in session_list.
    Also calculate running speed PSTH; the correlation between actual and virtual running speeds for openloop sessions.

    Args:
        flexilims_session (Series): flexilims session.
        session_list (list): list of session names.
        nbins (int, optional): number of bins for raster. Defaults to 10.
        closed_loop (int, optional): whether it's closedloop or openloop. Defaults to 1.
        use_cols (list, optional): list of useful columns. Defaults to [ "preferred_depth_closedloop_crossval", "depth_tuning_test_rsq_closedloop", ].
        rs_thr_min (float, optional): running speed min threshold. Defaults to None.
        rs_thr_max (float, optional): running speed max threshold. Defaults to None.
        still_only (bool, optional): whether to only take stationary frames. Defaults to False.
        still_time (float, optional): duration of stationary time. Defaults to 1.
        verbose (bool, optional): verbose. Defaults to 1.
        corridor_length (float, optional): corridor length for one trial. Defaults to 6.
        blank_length (float, optional): length of blank period at each end of the corridor. Defaults to 0.
        overwrite (bool, optional): whether to overwrite the existing results or not. Defaults to False.

    Returns:
        pd.DataFrame: concatenated neurons_df dataframe
    """
    results_all = []
    for isess, session_name in enumerate(session_list):
        print(f"{isess}/{len(session_list)}: calculating PSTH for {session_name}")
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            conflicts="skip",
        )
        psth_path = neurons_ds.path_full.parent / "psth_crossval.pkl"
        if psth_path.exists() and not overwrite:
            results_all.append(pd.read_pickle(psth_path))
            continue
        # Load all data
        if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5
        suite2p_ds = flz.get_datasets_recursively(
            flexilims_session=flexilims_session,
            origin_name=session_name,
            dataset_type="suite2p_traces",
        )
        fs = list(suite2p_ds.values())[0][-1].extra_attributes["fs"]
        try:
            neurons_df = pd.read_pickle(neurons_ds.path_full)
        except FileNotFoundError:
            print(f"ERROR: SESSION {session_name}: neurons_ds not found")
            continue
        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df
            else:
                neurons_df = neurons_df[use_cols]

            _, trials_df = spheres.sync_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets={"anatomical_only": 3},
                recording_type="two_photon",
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
            )
            trials_df = trials_df[trials_df.closed_loop == closed_loop]
            neurons_df["session"] = session_name
            # Add roi, preferred depth, iscell to results
            exp_session = flz.get_entity(
                datatype="session",
                name=session_name,
                flexilims_session=flexilims_session,
            )
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=exp_session.name,
                dataset_type="suite2p_rois",
                filter_datasets={"anatomical_only": 3},
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = np.load(
                suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True
            )[:, 0]
            neurons_df["iscell"] = iscell
            neurons_df["psth_crossval"] = [[np.nan]] * len(neurons_df)

            # Calculate dff psth crossval
            # Get the responses for this session that are not included for calculating the cross-validated preferred depth
            choose_trials_resp = list(
                set(neurons_df.depth_tuning_trials_closedloop.iloc[0])
                - set(neurons_df.depth_tuning_trials_closedloop_crossval.iloc[0])
            )
            trials_df_resp, _, _ = common_utils.choose_trials_subset(
                trials_df, choose_trials_resp, by_depth=True
            )

            print("Calculating dff PSTH")
            for roi in tqdm(range(len(neurons_df))):
                psth, _, _ = get_PSTH(
                    trials_df=trials_df_resp,
                    roi=roi,
                    is_closed_loop=closed_loop,
                    max_distance=corridor_length + blank_length,
                    min_distance=-blank_length,
                    nbins=nbins,
                    rs_thr_min=rs_thr_min,
                    rs_thr_max=rs_thr_max,
                    still_only=still_only,
                    still_time=still_time,
                    frame_rate=fs,
                    compute_ci=False,
                )
                neurons_df.at[roi, "psth_crossval"] = psth

            neurons_df.to_pickle(psth_path)
            results_all.append(neurons_df)
            if verbose:
                print(f"Finished concat neurons_df from session {session_name}")
        else:
            print(
                f"ERROR: SESSION {session_name}: specified cols not all in neurons_df"
            )
    results_all = pd.concat(results_all, axis=0, ignore_index=True)
    return results_all


def calculate_openloop_rs_correlation(
    imaging_df_openloop, trials_df, separate_depths=False
):
    if not separate_depths:
        rs_actual = imaging_df_openloop["RS"][
            (imaging_df_openloop["RS"].notnull())
            & (imaging_df_openloop["RS_eye"].notnull())
        ]
        rs_eye = imaging_df_openloop["RS_eye"][
            (imaging_df_openloop["RS"].notnull())
            & (imaging_df_openloop["RS_eye"].notnull())
        ]
        r_all, p_all = pearsonr(rs_actual, rs_eye)
    else:
        trials_df_openloop = trials_df[trials_df.closed_loop == 0]
        depth_list = find_depth_neurons.find_depth_list(trials_df)
        r_all = []
        p_all = []
        for depth in depth_list:
            rs_actual = np.hstack(
                trials_df_openloop[trials_df_openloop.depth == depth]["RS_stim"]
            )
            rs_eye = np.hstack(
                trials_df_openloop[trials_df_openloop.depth == depth]["RS_eye_stim"]
            )
            nan_vals = np.isnan(rs_actual) | np.isnan(rs_eye)
            rs_actual = rs_actual[~nan_vals]
            rs_eye = rs_eye[~nan_vals]
            r, p = pearsonr(rs_actual, rs_eye)
            r_all.append(r)
            p_all.append(p)
    return r_all, p_all


def get_rs_stats_all_sessions(
    flexilims_session,
    session_list,
    nbins=60,
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    corridor_length=6,
    blank_length=3,
    overwrite=False,
):
    """Calculate the PSTH for all sessions in session_list.
    Also calculate running speed PSTH; the correlation between actual and virtual running speeds for openloop sessions.

    Args:
        flexilims_session (Series): flexilims session.
        session_list (list): list of session names.
        nbins (int, optional): number of bins for raster. Defaults to 10.
        closed_loop (int, optional): whether it's closedloop or openloop. Defaults to 1.
        use_cols (list, optional): list of useful columns. Defaults to [ "preferred_depth_closedloop_crossval", "depth_tuning_test_rsq_closedloop", ].
        rs_thr_min (float, optional): running speed min threshold. Defaults to None.
        rs_thr_max (float, optional): running speed max threshold. Defaults to None.
        still_only (bool, optional): whether to only take stationary frames. Defaults to False.
        still_time (float, optional): duration of stationary time. Defaults to 1.
        verbose (bool, optional): verbose. Defaults to 1.
        corridor_length (float, optional): corridor length for one trial. Defaults to 6.
        blank_length (float, optional): length of blank period at each end of the corridor. Defaults to 0.
        overwrite (bool, optional): whether to overwrite the existing results or not. Defaults to False.

    Returns:
        pd.DataFrame: concatenated neurons_df dataframe
    """
    results_all = pd.DataFrame(
        columns=[
            [
                "session",
                "rs_psth_stim_closedloop",
                "rs_psth_closedloop",
                "rs_psth_for_of_closedloop",
                "rs_mean_trials_closedloop",
                "rs_mean_closedloop",
                "rs_psth_stim_openloop",
                "rs_psth_openloop",
                "rs_mean_trials_openloop",
                "rs_mean_openloop",
                "rs_correlation_rval_openloop",
                "rs_correlation_pval_openloop",
                "rs_correlation_rval_openloop_alldepths",
                "rs_correlation_pval_openloop_alldepths",
            ]
        ],
        index=np.arange(len(session_list)),
    )
    (
        results_all["rs_psth_stim_closedloop"],
        results_all["rs_psth_closedloop"],
        results_all["rs_psth_for_of_closedloop"],
        results_all["rs_mean_trials_closedloop"],
        results_all["rs_mean_closedloop"],
        results_all["rs_psth_stim_openloop"],
        results_all["rs_psth_openloop"],
        results_all["rs_mean_trials_openloop"],
        results_all["rs_mean_openloop"],
        results_all["rs_correlation_rval_openloop_alldepths"],
        results_all["rs_correlation_pval_openloop_alldepths"],
    ) = (
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
        [[np.nan]] * len(results_all),
    )
    for isess, session_name in enumerate(session_list):
        print(f"{isess}/{len(session_list)}: calculating RS stats for {session_name}")
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            conflicts="skip",
        )
        save_path = neurons_ds.path_full.parent / "rs_stats.pkl"
        if save_path.exists() and not overwrite:
            results_all.iloc[isess] = pd.read_pickle(save_path)
            continue
        # Load all data
        if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5
        suite2p_ds = flz.get_datasets_recursively(
            flexilims_session=flexilims_session,
            origin_name=session_name,
            dataset_type="suite2p_traces",
        )
        fs = list(suite2p_ds.values())[0][-1].extra_attributes["fs"]
        _, trials_df = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            filter_datasets={"anatomical_only": 3},
            recording_type="two_photon",
            protocol_base="SpheresPermTubeReward",
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )
        trials_df_original = trials_df.copy()
        for closed_loop in trials_df_original.closed_loop.unique():
            trials_df = trials_df_original[
                trials_df_original.closed_loop == closed_loop
            ]
            if closed_loop:
                sfx = "closedloop"
            else:
                sfx = "openloop"
            results_all.at[isess, "session"] = session_name

            # Calculate the running speed psth
            print("Calculating running speed PSTH")
            # just for stim period
            rs_psth_stim, _, _ = get_PSTH(
                trials_df=trials_df,
                roi=0,
                use_col="RS",
                is_closed_loop=closed_loop,
                max_distance=corridor_length,
                min_distance=0,
                nbins=nbins,
                rs_thr_min=rs_thr_min,
                rs_thr_max=rs_thr_max,
                still_only=still_only,
                still_time=still_time,
                frame_rate=fs,
                compute_ci=False,
            )
            results_all.at[isess, f"rs_psth_stim_{sfx}"] = np.expand_dims(
                rs_psth_stim, 0
            )

            # stim + some blank period
            rs_psth, _, _ = get_PSTH(
                trials_df=trials_df,
                roi=0,
                use_col="RS",
                is_closed_loop=closed_loop,
                max_distance=corridor_length + blank_length,
                min_distance=-blank_length,
                nbins=nbins,
                rs_thr_min=rs_thr_min,
                rs_thr_max=rs_thr_max,
                still_only=still_only,
                still_time=still_time,
                frame_rate=fs,
                compute_ci=False,
            )
            results_all.at[isess, f"rs_psth_{sfx}"] = np.expand_dims(rs_psth, 0)

            mean_rs = find_depth_neurons.average_dff_for_all_trials(
                trials_df=trials_df,
                use_col="RS_stim",
                rs_col="RS_stim",
                rs_thr=rs_thr_min,
                rs_thr_max=rs_thr_max,
                still_only=still_only,
                still_time=still_time,
                frame_rate=fs,
                closed_loop=closed_loop,
                param="depth",
            )
            results_all.at[isess, f"rs_mean_trials_{sfx}"] = np.expand_dims(mean_rs, 0)
            results_all.at[isess, f"rs_mean_{sfx}"] = np.expand_dims(
                np.expand_dims(np.mean(mean_rs, axis=1), 0), 0
            )

        # Calculate openloop rs and rs_eye correlation
        if len(trials_df_original.closed_loop.unique()) == 2:
            print("Calculating openloop RS correlation")
            _, imaging_df_openloop = spheres.regenerate_frames_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets={"anatomical_only": 3},
                recording_type="two_photon",
                is_closedloop=0,
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                resolution=5,
                do_regenerate_frames=False,
            )
            r, p = calculate_openloop_rs_correlation(
                imaging_df_openloop, trials_df_original, separate_depths=False
            )
            results_all.at[isess, "rs_correlation_rval_openloop"] = r
            results_all.at[isess, "rs_correlation_pval_openloop"] = p
            results_all.loc[isess, "rs_correlation_rval_openloop"] = results_all.loc[
                isess, "rs_correlation_rval_openloop"
            ].apply(lambda x: f"{x:.15e}")
            results_all.loc[isess, "rs_correlation_pval_openloop"] = results_all.loc[
                isess, "rs_correlation_pval_openloop"
            ].apply(lambda x: f"{x:.15e}")
            r_all, p_all = calculate_openloop_rs_correlation(
                imaging_df_openloop, trials_df_original, separate_depths=True
            )
            results_all.at[isess, "rs_correlation_rval_openloop_alldepths"] = (
                np.expand_dims(np.expand_dims(r_all, 0), 0)
            )
            results_all.at[isess, "rs_correlation_pval_openloop_alldepths"] = (
                np.expand_dims(np.expand_dims(p_all, 0), 0)
            )

        # append results_df
        # results_all.iloc[isess] = results_all.iloc[isess].apply(np.squeeze)
        results_all.iloc[isess].to_pickle(save_path)

    return results_all


def plot_preferred_depth_hist(
    results_df,
    use_col="preferred_depth_closedloop",
    nbins=50,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    results_df = results_df[results_df["iscell"] == 1].copy()
    # convert to cm
    results_df[use_col] = results_df[use_col].apply(lambda x: np.log(x * 100))
    min_depth = np.nanmin(results_df[use_col])
    max_depth = np.nanmax(results_df[use_col])
    depth_bins = np.linspace(
        min_depth,
        max_depth,
        num=nbins,
    )
    tol = 1e-4
    # set rows where use_col = min or max to -inf and inf
    results_df[use_col] = results_df[use_col].apply(
        lambda x: -np.inf if x < min_depth + tol else x
    )
    results_df[use_col] = results_df[use_col].apply(
        lambda x: np.inf if x > max_depth - tol else x
    )

    n, _, _ = plt.hist(
        results_df[use_col],
        bins=depth_bins,
        weights=np.ones(len(results_df)) / len(results_df),
        color="cornflowerblue",
        edgecolor="royalblue",
    )
    # plot proportion of rows with -inf and inf values as separate bars at min_depth/2 and max_depth*2
    plt.bar(
        min_depth - 1,
        np.sum(results_df[use_col] == -np.inf) / len(results_df),
        color="cornflowerblue",
        edgecolor="royalblue",
        width=(max_depth - min_depth) / nbins,
    )
    plt.bar(
        max_depth + 1,
        np.sum(results_df[use_col] == np.inf) / len(results_df),
        color="cornflowerblue",
        edgecolor="royalblue",
        width=(max_depth - min_depth) / nbins,
    )

    ax = plt.gca()
    ax.set_ylabel("Proportion of neurons", fontsize=fontsize_dict["label"])
    ax.set_xlabel("Preferred virtual depth (cm)", fontsize=fontsize_dict["label"])
    tick_pos = [10, 100, 1000]
    ax.set_xticks(
        np.log(
            np.concatenate(
                (
                    np.arange(2, 9, 1),
                    np.arange(2, 9, 1) * 10,
                    np.arange(2, 9, 1) * 100,
                    [
                        2000,
                    ],
                )
            )
        ),
        minor=True,
    )
    plt.xticks(
        np.concatenate([[min_depth - 1], np.log(tick_pos), [max_depth + 1]]),
        labels=np.concatenate([["N.P."], tick_pos, ["F.P."]]),
        fontsize=fontsize_dict["tick"],
    )

    plt.ylim([0, np.round(np.max(n), 2)])
    plt.yticks([0, np.round(np.max(n), 2)], fontsize=fontsize_dict["tick"])
    plotting_utils.despine()


def plot_psth_raster(
    results_df,
    depth_list,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    vmax=2,
):
    psths = np.stack(results_df["psth_crossval"])[:, :-1, 10:-10]  # exclude blank
    ndepths = psths.shape[1]
    nbins = psths.shape[2]
    # Sort neurons by preferred depth
    preferred_depths = results_df["preferred_depth_closedloop_crossval"].values
    psths = psths[preferred_depths.argsort()]
    psths = psths.reshape(psths.shape[0], -1)
    # zscore each row
    normed_psth = (psths - np.nanmean(psths, axis=1)[:, np.newaxis]) / (
        np.nanstd(psths, axis=1)[:, np.newaxis]
    )
    # Plot PSTHs
    ax = plt.gca()
    im = ax.imshow(
        normed_psth,
        aspect="auto",
        cmap="bwr",
        vmin=-vmax,
        vmax=vmax,
    )
    # Plot vertical lines to separate different depths
    for i in range(ndepths):
        ax.axvline((i + 1) * nbins, color="k", linewidth=0.5, linestyle="dotted")
    # Change xticks positions to the middle of current ticks and show depth at the tick position
    xticks = (np.arange(ndepths) + 0.5) * nbins
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(depth_list).astype("int"))
    ax.set_xlabel("Virtual depth (cm)", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="x", labelsize=fontsize_dict["tick"], rotation=60)
    ax.set_ylabel("Neuron number", fontsize=fontsize_dict["label"], labelpad=-5)
    ax.set_yticks([1, len(results_df)])
    ax.tick_params(axis="y", labelsize=fontsize_dict["tick"])
    ax.set_xlim([0, ndepths * nbins])

    # # for aligning with the scalebar
    # ax.vlines(1/4*60-10, -10, 9000, color="k", linestyle="--", linewidth=0.5)
    # ax.vlines(60-1/4*60-10, -10, 9000, color="k", linestyle="--", linewidth=0.5)

    ax_pos = ax.get_position()
    ax2 = plt.gcf().add_axes(
        [
            ax_pos.x1 + ax_pos.width * 0.05,
            ax_pos.y0,
            0.01,
            ax_pos.height / 2,
        ]
    )
    cbar = plt.colorbar(mappable=im, cax=ax2)
    cbar.set_label("Z-score", fontsize=fontsize_dict["legend"])
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"])


def plot_depth_neuron_perc_hist(
    results_df,
    bins=50,
    ylim=None,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    session_prop = results_df.groupby("session").agg({"depth_tuned": "mean"})
    plt.hist(
        session_prop["depth_tuned"],
        bins=bins,
        color="cornflowerblue",
        edgecolor="royalblue",
    )
    ax = plt.gca()
    xlim = ax.get_xlim()
    ax.set_xlim([0, xlim[1]])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(
        "Proportion of \ndepth-tuned neurons", fontsize=fontsize_dict["label"]
    )
    ax.set_ylabel("Number of sessions", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])
    # plot median proportion as a triangle along the top of the histogram
    median_prop = np.median(session_prop["depth_tuned"])
    print("Median proportion of depth-tuned neurons:", median_prop)
    print(
        "Range of proportions of depth-tuned neurons:",
        np.min(session_prop["depth_tuned"]),
        "to",
        np.max(session_prop["depth_tuned"]),
    )
    print("Number of sessions:", len(session_prop))
    ax.plot(
        median_prop,
        ax.get_ylim()[1] * 0.95,
        marker="v",
        markersize=5,
        markerfacecolor="cornflowerblue",
        markeredgecolor="royalblue",
    )
    plotting_utils.despine()


def get_visually_responsive_neurons(
    trials_df, neurons_df, is_closed_loop=1, before_onset=0.5, frame_rate=15
):
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    # Find the mean response of each trial for all ROIs
    trials_df["trial_mean_response"] = trials_df.apply(
        lambda x: np.mean(x.dff_stim, axis=0), axis=1
    )

    # Find the mean response of the blank period before the next trial for all ROIs
    trials_df["trial_mean_onset"] = trials_df.apply(
        lambda x: np.mean(x.dff_blank[-int(frame_rate * before_onset) :], axis=0),
        axis=1,
    )
    # Shift blank response down 1 to put it to the correct trial
    trials_df["trial_mean_onset"] = trials_df["trial_mean_onset"].shift(1)

    all_response = np.stack(trials_df.trial_mean_response[1:].values)
    all_onset = np.stack(trials_df.trial_mean_onset[1:].values)

    # Check whether the response is significantly higher than the blank period
    for iroi, roi in enumerate(neurons_df.roi):
        response = all_response[:, iroi]
        onset = all_onset[:, iroi]
        pval = scipy.stats.wilcoxon(response, onset).pvalue
        neurons_df.at[roi, "visually_responsive"] = (pval < 0.05) & (
            np.mean(response - onset) > 0
        )
        neurons_df.at[roi, "visually_responsive_pval"] = pval
        neurons_df.at[roi, "mean_resp"] = np.mean(response - onset)

    return neurons_df


def get_visually_responsive_all_sessions(
    flexilims_session,
    session_list,
    use_cols,
    is_closed_loop=1,
    protocol_base="SpheresPermTubeReward",
    protocol_base_list=[],
    before_onset=0.5,
    frame_rate=15,
):
    isess = 0
    for i, session_name in enumerate(session_list):
        print(f"Calculating visually responsive neurons for {session_name}")
        if len(protocol_base_list) > 0:
            protocol_base = protocol_base_list[i]

        # Load all data
        if ("PZAH6.4b" in session_name) | ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5

        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        _, trials_df = spheres.sync_all_recordings(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=None,
            filter_datasets={"anatomical_only": 3},
            recording_type="two_photon",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )
        trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
        neurons_df = get_visually_responsive_neurons(
            trials_df,
            neurons_df,
            is_closed_loop=is_closed_loop,
            before_onset=before_onset,
            frame_rate=frame_rate,
        )
        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df
            else:
                neurons_df = neurons_df[use_cols]

            neurons_df["session"] = session_name
            exp_session = flz.get_entity(
                datatype="session",
                name=session_name,
                flexilims_session=flexilims_session,
            )
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=exp_session.name,
                dataset_type="suite2p_rois",
                filter_datasets={"anatomical_only": 3},
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = np.load(
                suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True
            )[:, 0]
            neurons_df["iscell"] = iscell

            if isess == 0:
                results_all = neurons_df.copy()
            else:
                results_all = pd.concat(
                    [results_all, neurons_df], axis=0, ignore_index=True
                )
            isess += 1
        else:
            print(
                f"ERROR: SESSION {session_name}: specified cols not all in neurons_df"
            )

    return results_all


def get_color(value, value_min, value_max, log=False, cmap=cm.cool.reversed()):
    if log:
        norm = matplotlib.colors.Normalize(
            vmin=np.log(value_min), vmax=np.log(value_max)
        )
        rgba_color = cmap(norm(np.log(value)), bytes=True)
    else:
        norm = matplotlib.colors.Normalize(vmin=value_min, vmax=value_max)
        rgba_color = cmap(norm(value), bytes=True)
    rgba_color = np.array([it / 255 for it in rgba_color])

    return rgba_color


def plot_example_fov(
    neurons_df,
    stat,
    ops,
    ndepths=8,
    col="preferred_depth",
    cmap=cm.cool.reversed(),
    background_color=np.array([0.133, 0.545, 0.133]),
    n_std=6,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    fov_width=572.867,
):
    rf.find_rf_centers(
        neurons_df,
        ndepths=ndepths,
        frame_shape=(16, 24),
        is_closed_loop=1,
        resolution=5,
    )
    find_roi_centers(neurons_df, stat)
    if col == "preferred_depth":
        select_neurons = neurons_df[
            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
            & (neurons_df["iscell"] == 1)
        ]
        null_neurons = neurons_df[
            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] >= 0.05)
            & (neurons_df["iscell"] == 1)
        ]
    else:
        coef = np.stack(neurons_df[f"rf_coef_closedloop"].values)
        coef_ipsi = np.stack(neurons_df[f"rf_coef_ipsi_closedloop"].values)
        sig, _ = spheres.find_sig_rfs(
            np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
            np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
            n_std=n_std,
        )
        select_neurons = neurons_df[(sig == 1) & (neurons_df["iscell"] == 1)]
        null_neurons = neurons_df[(sig == 0) & (neurons_df["iscell"] == 1)]
    # Find neuronal masks and assign on the background image
    im = (
        np.ones((ops["Ly"], ops["Lx"], 3)) * background_color[np.newaxis, np.newaxis, :]
    )
    azi_min = select_neurons["rf_azi"].quantile(0.1)
    azi_max = select_neurons["rf_azi"].quantile(0.9)
    ele_min = select_neurons["rf_ele"].quantile(0.1)
    ele_max = select_neurons["rf_ele"].quantile(0.9)
    for i, n in null_neurons.iterrows():
        ypix = stat[n.roi]["ypix"][~stat[n.roi]["overlap"]]
        xpix = stat[n.roi]["xpix"][~stat[n.roi]["overlap"]]
        if len(xpix) > 0 and len(ypix) > 0:
            im[ypix, xpix, :] = 0.3
    for _, n in select_neurons.iterrows():
        ypix = stat[n.roi]["ypix"][~stat[n.roi]["overlap"]]
        xpix = stat[n.roi]["xpix"][~stat[n.roi]["overlap"]]
        if col == "preferred_depth_closedloop":
            rgba_color = get_color(
                n[col],
                0.02,
                20,
                log=True,
                cmap=cmap,
            )
        elif col == "rf_azi":
            rgba_color = get_color(
                n[col],
                azi_min,
                azi_max,
                log=False,
                cmap=cmap,
            )
        elif col == "rf_ele":
            rgba_color = get_color(
                n[col],
                ele_min,
                ele_max,
                log=False,
                cmap=cmap,
            )
        im[ypix, xpix, :] = rgba_color[np.newaxis, :3]
    # Plot spatial distribution
    plt.imshow(im)
    if col != "preferred_depth_closedloop":
        # find the gradient of col w.r.t. center_x and center_y
        slope_x = scipy.stats.linregress(
            x=select_neurons["center_x"], y=select_neurons[col]
        ).slope
        slope_y = scipy.stats.linregress(
            x=select_neurons["center_y"], y=select_neurons[col]
        ).slope
        norm = np.linalg.norm(np.array([slope_x, slope_y]))
        slope_x /= norm
        slope_y /= norm
        arrow_length = 100
        # draw an arrow in the direction of the gradient
        plt.arrow(
            x=im.shape[1] / 2 - slope_x * arrow_length / 2,
            y=im.shape[0] / 2 - slope_y * arrow_length / 2,
            dx=slope_x * arrow_length,
            dy=slope_y * arrow_length,
            color="white",
            width=2,
            head_width=50,
        )
        print(f"{col}: slope x {slope_x}, slope y {slope_y}")
    plt.axis("off")
    # Add a colorbar for the dummy plot with the new colormap
    cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=plt.gca())
    cbar.set_ticks(np.linspace(0, 1, 3))
    if col == "preferred_depth_closedloop":
        cbar.set_ticklabels((np.geomspace(0.02, 20, 3) * 100).astype("int"))
    elif col == "rf_azi":
        cbar.set_ticklabels(np.round(np.linspace(azi_min, azi_max, 3), 1))
    elif col == "rf_ele":
        cbar.set_ticklabels(np.round(np.linspace(ele_min, ele_max, 3), 1))
    cbar.ax.tick_params(labelsize=fontsize_dict["legend"])
    cbar_pos = np.array(plt.gca().get_position().bounds)
    cbar_pos[0] = cbar_pos[0] + cbar_pos[2] + 0.01
    cbar_pos[2] = 0.25
    cbar_pos[3] = cbar_pos[3] * 0.3
    cbar.ax.set_position(cbar_pos)
    cbar.ax.tick_params(axis="y", length=1.5)
    # Add scalebar
    scalebar_length_px = im.shape[0] / fov_width * 100  # Scale bar length in pixels
    rect = plt.Rectangle(
        (800, im.shape[0] * 0.95),
        scalebar_length_px,
        scalebar_length_px * 0.05,
        color="white",
    )
    plt.gca().invert_xaxis()
    plt.gca().add_patch(rect)
    return im


def plot_fov_mean_img(im, vmax=700, fov_width=572.867):
    plt.imshow(np.flip(im, axis=1), vmax=vmax, cmap="gray")
    plt.axis("off")
    cbar = plt.colorbar()
    cbar_pos = np.array(plt.gca().get_position().bounds)
    cbar_pos[0] = cbar_pos[0] + cbar_pos[2] + 0.005
    cbar_pos[2] = 0.15
    cbar_pos[3] = cbar_pos[3] * 0.3
    cbar.ax.set_position(cbar_pos)
    cbar.ax.tick_params(axis="y", length=1.5)
    cbar.remove()
    # Add scalebar
    scalebar_length_px = im.shape[0] / fov_width * 100  # Scale bar length in pixels
    rect = plt.Rectangle(
        (40, im.shape[0] * 0.93), scalebar_length_px, 20, color="white"
    )
    plt.gca().add_patch(rect)


def plot_running_stationary_depth_tuning(
    roi,
    roi_num,
    i,
    neurons_df,
    trials_df,
    ax,
    depth_tuning_kwargs,
    fontsize_dict,
    fov_ax=None,
    ops=None,
    stat=None,
    legend_loc="upper right",
    text_pos="upper_left",
):
    ylims = []
    for (
        rs_thr,
        rs_thr_max,
        still_only,
        still_time,
        i_running,
        linecolor,
        label,
        use_col,
    ) in zip(
        [0.05, None],
        [None, 0.05],
        [0, 1],
        [0, 1],
        [0, 1],
        ["royalblue", "gray"],
        ["Running", "Stationary"],
        [
            "depth_tuning_popt_closedloop_running",
            "depth_tuning_popt_closedloop_notrunning",
        ],
    ):
        # calculate ylim
        mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(
            trials_df=trials_df,
            rs_thr=rs_thr,
            rs_thr_max=rs_thr_max,
            still_only=still_only,
            still_time=still_time,
            frame_rate=15,
            closed_loop=1,
            param="depth",
        )[:, :, roi]
        CI_low, CI_high = common_utils.get_bootstrap_ci(mean_dff_arr)
        ylim = (np.nanmin(CI_low), np.nanmax(CI_high))
        ylims.append(ylim)

    ylim = (
        min([i[0] for i in ylims]),
        plt_common_utils.ceil(max([i[1] for i in ylims]), 1),
    )
    # plot
    for (
        rs_thr,
        rs_thr_max,
        still_only,
        still_time,
        i_running,
        linecolor,
        label,
        use_col,
    ) in zip(
        [0.05, None],
        [None, 0.05],
        [0, 1],
        [0, 1],
        [0, 1],
        ["royalblue", "gray"],
        ["Running", "Stationary"],
        [
            "depth_tuning_popt_closedloop_running",
            "depth_tuning_popt_closedloop_notrunning",
        ],
    ):
        depth_tuning_running_kwargs = depth_tuning_kwargs.copy()
        depth_tuning_running_kwargs["rs_thr"] = rs_thr
        depth_tuning_running_kwargs["rs_thr_max"] = rs_thr_max
        depth_tuning_running_kwargs["still_only"] = still_only
        depth_tuning_running_kwargs["still_time"] = still_time
        depth_tuning_running_kwargs["linecolor"] = linecolor
        depth_tuning_running_kwargs["use_col"] = use_col
        plot_depth_tuning_curve(
            neurons_df=neurons_df,
            trials_df=trials_df,
            roi=roi,
            **depth_tuning_running_kwargs,
            ylim=ylim,
            label=label,
        )

        if i != 1:
            plt.ylabel("")
        if (i % 3 != 2) and roi_num != 1:
            plt.xlabel("")
            ax.set_xticklabels([])
        if i_running == 0:
            if text_pos == "upper_left":
                x_label = plt.xlim()[0] + 0.05 * (plt.xlim()[1] - plt.xlim()[0])
            elif text_pos == "upper_right":
                x_label = plt.xlim()[1] - 0.3 * (plt.xlim()[1] - plt.xlim()[0])
            plt.text(
                x_label,
                ylim[1],
                f"Cell {roi_num}",
                fontsize=fontsize_dict["legend"],
            )
            if fov_ax:
                fov_ax.annotate(
                    f"{roi_num}",
                    (
                        ops["meanImg"].shape[0] - stat[roi]["med"][1],
                        stat[roi]["med"][0],
                    ),
                    xytext=(5, 5),
                    textcoords="offset points",
                    color="w",
                    fontsize=fontsize_dict["label"],
                    arrowprops=dict(facecolor="w", edgecolor="w", arrowstyle="->"),
                )
    if i == 0:
        plt.legend(
            loc=legend_loc,
            fontsize=fontsize_dict["legend"],
            framealpha=1,
            borderpad=0,
            frameon=False,
            handlelength=0.5,
        )


def plot_mean_running_speed_alldepths(
    results,
    depth_list,
    fontsize_dict,
    param="RS",
    ylim=None,
    of_threshold=0.01,
    linewidth=3,
    elinewidth=3,
    jitter=0.2,
    scatter_markersize=2,
    scatter_alpha=0.5,
    capsize=3,
    capthick=10,
):
    ax = plt.gca()
    if param == "RS":
        rs_means = (
            np.vstack([j for i in results.rs_mean_closedloop.values for j in i]) * 100
        )
    elif param == "OF":
        rs_means = np.degrees(
            np.vstack([j for i in results.rs_mean_closedloop.values for j in i])
            / depth_list.reshape(1, -1)
        )
        rs_means[rs_means < of_threshold] = of_threshold
    CI_low, CI_high = common_utils.get_bootstrap_ci(rs_means.T, sig_level=0.05)
    for idepth in range(len(depth_list)):
        color = plotting_utils.get_color(
            depth_list[idepth], 
            value_min=np.min(depth_list),
            value_max=np.max(depth_list),
            log=True,
            cmap=cm.cool.reversed()
        )
        sns.stripplot(
            x=np.ones(rs_means.shape[0]) * idepth,
            y=rs_means[:, idepth],
            jitter=jitter,
            edgecolor="white",
            color=color,
            alpha=scatter_alpha,
            size=scatter_markersize,
        )
        plt.plot(
            [idepth - 0.4, idepth + 0.4],
            [np.mean(rs_means[:, idepth]), np.mean(rs_means[:, idepth])],
            linewidth=linewidth,
            color=color,
        )
        plt.errorbar(
            x=idepth,
            y=np.mean(rs_means[:, idepth]),
            yerr=np.array(
                [
                    np.mean(rs_means[:, idepth]) - CI_low[idepth],
                    CI_high[idepth] - np.mean(rs_means[:, idepth]),
                ]
            ).reshape(2, 1),
            capsize=capsize,
            elinewidth=elinewidth,
            ecolor=color,
            capthick=capthick,
        )
    ax.set_xticklabels(
        np.round((depth_list * 100)).astype("int"), fontsize=fontsize_dict["label"]
    )
    if param == "RS":
        ax.set_ylabel("Average running\nspeed (cm/s)", fontsize=fontsize_dict["label"])
        ax.set_ylim(0, ax.get_ylim()[1])
    elif param == "OF":
        ax.set_ylabel(
            "Average optic flow\nspeed (degrees/s)", fontsize=fontsize_dict["label"]
        )
        ax.set_yscale("log")
    ax.set_xlabel("Depth (cm)", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    if ylim is not None:
        ax.set_ylim(ylim)
        if param == "RS":
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 4))
    sns.despine(ax=ax)
