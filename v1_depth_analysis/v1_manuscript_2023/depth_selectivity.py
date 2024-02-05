import functools
print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pickle
from tqdm import tqdm
import scipy

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import spheres, find_depth_neurons, common_utils, fit_gaussian_blob, size_control
from cottage_analysis.plotting import basic_vis_plots, plotting_utils
from cottage_analysis.pipelines import pipeline_utils


def plot_raster_all_depths(
    fig, 
    neurons_df,
    trials_df,
    roi,
    is_closed_loop,
    max_distance=6,
    nbins=60,
    frame_rate=15,
    vmax=1,
    plot=True,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.05,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    """Raster plot for neuronal activity for each depth

    Args:
        fig (plt.figure): figure to plot on.
        neurons_df (pd.DataFrame): dataframe with analyzed info of all rois.
        trials_df (pd.DataFrame): dataframe with info of all trials.
        roi (int): ROI number
        is_closed_loop (bool): plotting the closed loop or open loop results.
        max_distance (int, optional): max distance for each trial in meters. Defaults to 6.
        nbins (int, optional): number of bins to bin the activity. Defaults to 60.
        frame_rate (int, optional): imaging frame rate. Defaults to 15.
        vmax (int, optional): vmax to plot the heatmap. Defaults to 1.
        plot (bool, optional): whether to plot or not. Defaults to True.
        plot_x (int, optional): x position of the plot out of 1. Defaults to 0.
        plot_y (int, optional): y position of the plot out of 1. Defaults to 0.
        plot_width (int, optional): width of the plot out of 1. Defaults to 1.
        plot_height (int, optional): height of the plot out of 1. Defaults to 1.
        cbar_width (float, optional): width of the colorbar. Defaults to 0.05.
        fontsize_dict (dict, optional): dictionary of fontsize for title, label and tick. Defaults to {"title": 20, "label": 15, "tick": 15}.
    """
    # choose the trials with closed or open loop to visualize
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")
    trial_number = len(trials_df) // len(depth_list)

    # bin dff according to distance travelled for each trial
    dffs_binned = np.zeros((len(depth_list), trial_number, nbins))
    for idepth, depth in enumerate(depth_list):
        all_dffs = grouped_trials.get_group(depth).dff_stim.values
        all_rs = grouped_trials.get_group(depth).RS_stim.values
        for itrial in np.arange(len(all_dffs)):
            if itrial < trial_number:
                dff = all_dffs[itrial][:, roi]
                rs_arr = all_rs[itrial]
                distance = np.cumsum(rs_arr / frame_rate)
                bins = np.linspace(
                    start=0, stop=max_distance, num=nbins + 1, endpoint=True
                )
                bin_means, _, _ = scipy.stats.binned_statistic(
                    x=distance,
                    values=dff,
                    statistic="mean",
                    bins=nbins,
                )
                dffs_binned[idepth, itrial, :] = bin_means

    # colormap
    WhRdcmap = basic_vis_plots.generate_cmap(cmap_name="WhRd")

    # plot each depth as a heatmap
    if plot:
        plot_prop = 0.75
        each_plot_width = (plot_width-cbar_width)/ len(depth_list)
        for idepth, depth in enumerate(depth_list):
            ax = fig.add_axes([plot_x+idepth*each_plot_width, plot_y, each_plot_width*plot_prop, plot_height]) 
            im = ax.imshow(
                dffs_binned[idepth], aspect="auto", cmap=WhRdcmap, vmin=0, vmax=vmax
            )
            plt.xticks(
                np.linspace(0, nbins, 3),
                (np.linspace(0, max_distance, 3) * 100).astype("int"),
            )
            if idepth == 0:
                ax.set_ylabel("Trial no.", fontsize=fontsize_dict["label"])
                ax.tick_params(
                    left=True,
                    right=False,
                    labelleft=True,
                    labelbottom=True,
                    bottom=True,
                    labelsize=fontsize_dict["tick"],
                )
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.tick_params(
                    left=True,
                    right=False,
                    labelleft=False,
                    labelbottom=True,
                    bottom=True,
                    labelsize=fontsize_dict["tick"],
                )
                ax.tick_params(axis="x", rotation=45)
            if idepth == len(depth_list)//2:
                ax.set_xlabel("Virtual distance (cm)", fontsize=fontsize_dict["label"])

        ax2 = fig.add_axes([plot_x + (len(depth_list)-1)*each_plot_width + each_plot_width*plot_prop + 0.01, plot_y, cbar_width*0.8, plot_height]) 
        fig.colorbar(im, cax=ax2, label="\u0394F/F")
                
    return dffs_binned


def plot_depth_tuning_curve(
    fig,
    neurons_df,
    trials_df,
    roi,
    param="depth",
    use_col="depth_tuning_popt_closedloop",
    rs_thr=None,
    rs_thr_max=None,
    still_only=False,
    still_time=0,
    frame_rate=15,
    plot_fit=True,
    linewidth=3,
    linecolor="k",
    fit_linecolor="r",
    closed_loop=1,
    overwrite_ax=True,
    ax=None,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    label=None,
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
    trials_df = size_control.get_physical_size(trials_df, use_cols=["size", "depth"], k=1)
    if param == "depth":
        param_list = np.array(find_depth_neurons.find_depth_list(trials_df))
    elif param == "size":
        param_list = np.sort(trials_df["physical_size"].unique())
    mean_dff_arr = find_depth_neurons.average_dff_for_all_trials(trials_df=trials_df,
                                                rs_thr=rs_thr, 
                                                rs_thr_max=rs_thr_max, 
                                                still_only=still_only, 
                                                still_time=still_time, 
                                                frame_rate=frame_rate, 
                                                closed_loop=closed_loop, 
                                                param=param)[:, :, roi]
    CI_low, CI_high = common_utils.get_confidence_interval(mean_dff_arr)
    mean_arr = np.nanmean(mean_dff_arr, axis=1)

    # Load gaussian fit params for this roi
    if plot_fit:
        min_sigma = 0.5
        [a, x0, log_sigma, b] = neurons_df.loc[roi, use_col]
        x = np.geomspace(param_list[0], param_list[-1], num=100)
        gaussian_arr = find_depth_neurons.gaussian_func(
            np.log(x), a, x0, log_sigma, b, min_sigma
        )

    # Plotting
    if overwrite_ax:
        ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
    ax.plot(np.log(param_list), mean_arr, color=linecolor, linewidth=linewidth, label=label)
    ax.fill_between(
        np.log(param_list),
        CI_low,
        CI_high,
        color=linecolor,
        alpha=0.3,
        edgecolor=None,
        rasterized=False,
    )
    if plot_fit:
        ax.plot(np.log(x), gaussian_arr, color=fit_linecolor, linewidth=linewidth)
    if param=="depth":
        plt.xticks(
            np.log(param_list),
            (np.array(param_list) * 100).astype("int"),
            fontsize=fontsize_dict["tick"],
            rotation=45,
        )
        plt.xlabel(f"Virtual depth (cm)", fontsize=fontsize_dict["label"])
    elif param=="size":
        plt.xticks(
            np.log(param_list),
            np.round(np.array(param_list)*0.87/10*20,1),
            fontsize=fontsize_dict["tick"],
            rotation=45,
        )
        plt.xlabel(f"Actual radius (cm)", fontsize=fontsize_dict["label"])
    plt.yticks(fontsize=fontsize_dict["tick"])
    ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"])

    plotting_utils.despine()
    
    return ax
    
    
def get_PSTH(
    trials_df,
    roi,
    is_closed_loop,
    rs_thr_min=None, #m/s
    rs_thr_max=None, #m/s
    still_only=False,
    still_time=1, #s
    max_distance=6, #m
    nbins=20,
    frame_rate=15,
):
    # confidence interval z calculation
    ci_range = 0.95
    z = scipy.stats.norm.ppf(1 - ((1 - ci_range) / 2))

    # choose the trials with closed or open loop to visualize
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]

    depth_list = find_depth_neurons.find_depth_list(trials_df)
    grouped_trials = trials_df.groupby(by="depth")
    trial_number = len(trials_df) // len(depth_list)

    # bin dff according to distance travelled for each trial
    all_means = np.zeros(((len(depth_list) + 1), nbins))
    all_ci = np.zeros(((len(depth_list) + 1), nbins))
    for idepth, depth in enumerate(depth_list):
        all_dff = []
        all_distance = []
        for itrial in np.arange(trial_number):
            dff = grouped_trials.get_group(depth).dff_stim.values[itrial][:, roi]
            rs_arr = grouped_trials.get_group(depth).RS_stim.values[itrial]

            # threshold running speed according to rs_thr 
            if not still_only: # take running frames
                if (rs_thr_min is None) and (rs_thr_max is None): # take all frames
                    take_idx = np.arange(len(rs_arr))
                else:
                    if rs_thr_max is None: # take frames with running speed > rs_thr
                        take_idx = rs_arr > rs_thr_min
                    elif rs_thr_min is None: # take frames with running speed < rs_thr
                        take_idx = rs_arr < rs_thr_max
                    else: # take frames with running speed between rs_thr_min and rs_thr_max
                        take_idx = (rs_arr > rs_thr_min) & (rs_arr < rs_thr_max)
            else: # take still frames
                if rs_thr_max is None: # use not running data but didn't set rs_thr
                    print("ERROR: calculating under not_running condition without rs_thr_max to determine max speed")
                else: # take frames with running speed < rs_thr for x seconds
                    take_idx = common_utils.find_thresh_sequence(array=rs_arr, threshold_max=rs_thr_max, length=int(still_time*frame_rate), shift=int(still_time*frame_rate))
                    
            dff = dff[take_idx]
            rs_arr = rs_arr[take_idx]
            distance = np.cumsum(rs_arr / frame_rate)
            all_dff.append(dff)
            all_distance.append(distance)

        all_dff = np.array([j for i in all_dff for j in i])
        all_distance = np.array([j for i in all_distance for j in i])
        bins = np.linspace(start=0, stop=max_distance, num=nbins + 1, endpoint=True)
        bin_centers = (bins[1:] + bins[:-1]) / 2

        # calculate speed tuning
        bin_means, _, _ = scipy.stats.binned_statistic(
            x=all_distance,
            values=all_dff,
            statistic="mean",
            bins=nbins,
        )

        bin_stds, _, _ = scipy.stats.binned_statistic(
            x=all_distance,
            values=all_dff,
            statistic="std",
            bins=nbins,
        )

        bin_counts, _, _ = scipy.stats.binned_statistic(
            x=all_distance,
            values=all_dff,
            statistic="count",
            bins=nbins,
        )

        all_means[idepth, :] = bin_means

        ci = z * bin_stds / np.sqrt(bin_counts)
        ci[np.isnan(ci)] = 0
        all_ci[idepth, :] = ci

    # Blank dff
    dff = trials_df.dff_blank.values
    rs = trials_df.RS_blank.values

    all_dff = np.array([j for i in dff for j in i[:, roi]])
    rs_arr = np.array([j for i in rs for j in i])
    all_distance = np.cumsum(rs_arr / frame_rate)

    # calculate speed tuning
    bin_means, _, _ = scipy.stats.binned_statistic(
        x=all_distance,
        values=all_dff,
        statistic="mean",
        bins=nbins,
    )

    bin_stds, _, _ = scipy.stats.binned_statistic(
        x=all_distance,
        values=all_dff,
        statistic="std",
        bins=nbins,
    )

    bin_counts, _, _ = scipy.stats.binned_statistic(
        x=all_distance,
        values=all_dff,
        statistic="count",
        bins=nbins,
    )

    all_means[-1, :] = bin_means
    ci = z * bin_stds / np.sqrt(bin_counts)
    ci[np.isnan(ci)] = 0
    all_ci[-1, :] = ci
    
    return all_means, all_ci, bin_centers
    

def plot_PSTH(
    fig,
    trials_df,
    roi,
    is_closed_loop,
    max_distance=6,
    nbins=20,
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    frame_rate=15,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    linewidth=3,
    legend_on=False,
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
    all_means, all_ci, bin_centers = get_PSTH(trials_df=trials_df, 
                                              roi=roi, 
                                              is_closed_loop=is_closed_loop, 
                                              rs_thr_min=rs_thr_min,
                                              rs_thr_max=rs_thr_max,
                                              still_only=still_only,
                                              still_time=still_time,
                                              max_distance=max_distance, 
                                              nbins=nbins, 
                                              frame_rate=frame_rate)
    
    depth_list = find_depth_neurons.find_depth_list(trials_df)  
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
    for idepth, depth in enumerate(depth_list):
        linecolor = basic_vis_plots.get_depth_color(depth, depth_list, cmap=cm.cool.reversed())
        ax.plot(
            bin_centers,
            all_means[idepth, :],
            color=linecolor,
            label=f"{int(depth_list[idepth] * 100)} cm",
            linewidth=linewidth,
        )

        ax.fill_between(
            bin_centers,
            y1=all_means[idepth, :] - all_ci[idepth, :],
            y2=all_means[idepth, :] + all_ci[idepth, :],
            color=linecolor,
            alpha=0.3,
            edgecolor=None,
            rasterized=False,
        )

    ax.plot(
        bin_centers,
        all_means[-1, :],
        color="gray",
        label=f"{int(depth_list[idepth] * 100)} cm",
        linewidth=linewidth,
    )
    ax.fill_between(
        bin_centers,
        y1=all_means[-1, :] - all_ci[-1, :],
        y2=all_means[-1, :] + all_ci[-1, :],
        color="gray",
        alpha=0.3,
        edgecolor=None,
        rasterized=False,
    )

    ax.set_xlabel("Virtual distance (m)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
    plt.xticks(
        # np.linspace(0, nbins, 3),
        # (np.linspace(0, max_distance, 3) * 100).astype("int"),
        fontsize=fontsize_dict["tick"],
        rotation=45,
    )
    plt.yticks(fontsize=fontsize_dict["tick"])
    
    if legend_on:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize_dict["legend"], frameon=False)
    plotting_utils.despine()
    
    
def get_psth_crossval_all_sessions(flexilims_session, session_list, nbins=10, closed_loop=1, use_cols=["preferred_depth_closedloop_crossval","depth_tuning_test_rsq_closedloop"], rs_thr_min=None, rs_thr_max=None, still_only=False, still_time=1, verbose=1):
    for isess, session_name in enumerate(session_list):
        print(f"Calculating PSTH for {session_name}")
        
        # Load all data
        if ("PZAH6.4b" or "PZAG3.4f") in session_name:
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5
            
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name, flexilims_session=flexilims_session, project=None, conflicts="skip"
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        if (use_cols is None) or (set(use_cols).issubset(neurons_df.columns.tolist())):
            if use_cols is None:
                neurons_df = neurons_df
            else:
                neurons_df = neurons_df[use_cols]
        
            _, trials_df = spheres.sync_all_recordings(
                session_name=session_name,
                flexilims_session=flexilims_session,
                project=None,
                filter_datasets={'anatomical_only':3},
                recording_type="two_photon",
                protocol_base="SpheresPermTubeReward",
                photodiode_protocol=photodiode_protocol,
                return_volumes=True,
                )
            trials_df = trials_df[trials_df.closed_loop==closed_loop]

            neurons_df['session'] = session_name
            
            # Create dataframe to store results
            results = pd.DataFrame(
            columns = [
                'session',
                'roi',
                'iscell',
                'preferred_depth_crossval',
                'preferred_depth_rsq',
                'psth_crossval',
                ]   
        )
            # Add roi, preferred depth, iscell to results
            results["roi"] = np.arange(len(neurons_df))
            results["session"] = session_name
            results["preferred_depth_crossval"] = neurons_df["preferred_depth_closedloop_crossval"]
            results["preferred_depth_rsq"] = neurons_df["depth_tuning_test_rsq_closedloop"]
            exp_session = flz.get_entity(
                datatype="session", name=session_name, flexilims_session=flexilims_session
            )
            suite2p_ds = flz.get_datasets(
                flexilims_session=flexilims_session,
                origin_name=exp_session.name,
                dataset_type="suite2p_rois",
                filter_datasets={"anatomical_only": 3},
                allow_multiple=False,
                return_dataseries=False,
            )
            iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[:,0]
            results["iscell"] = iscell
            results["psth_crossval"] = [[np.nan]]*len(neurons_df)
        
            # Get the responses for this session that are not included for calculating the cross-validated preferred depth
            choose_trials_resp = list(set(neurons_df.depth_tuning_trials_closedloop.iloc[0])-set(neurons_df.depth_tuning_trials_closedloop_crossval.iloc[0]))
            trials_df_resp, _, _ = common_utils.choose_trials_subset(trials_df, choose_trials_resp)
            
            for roi in tqdm(range(len(neurons_df))):
                psth, _, _ = get_PSTH(trials_df=trials_df_resp, 
                                                        roi=roi, 
                                                        is_closed_loop=1, 
                                                        max_distance=6, 
                                                        nbins=nbins, 
                                                        rs_thr_min=rs_thr_min,
                                                        rs_thr_max=rs_thr_max,
                                                        still_only=still_only,
                                                        still_time=still_time,
                                                        frame_rate=15)
                results.at[roi, "psth_crossval"] = psth
            
            if isess == 0:
                results_all = results.copy()
            else:
                results_all = pd.concat([results_all, results], axis=0, ignore_index=True)
            
            if verbose:
                print(f"Finished concat neurons_df from session {session_name}")
        else:
            print(f"ERROR: SESSION {session_name}: specified cols not all in neurons_df")
        
    return results_all


def plot_preferred_depth_hist(
    fig,
    results_df,
    use_col="preferred_depth_crossval",
    nbins=50,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    results_df = results_df[results_df['iscell'] ==1]
    depth_bins = np.geomspace(np.nanmin(results_df[use_col])*100,np.nanmax(results_df[use_col])*100,num=nbins)
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
    n, _, _ = ax.hist(results_df[use_col]*100, bins=depth_bins, weights=np.ones(len(results_df)) / len(results_df), color='k')
    ax.set_xscale('log')
    ax.set_ylabel('Proportion of neurons', fontsize = fontsize_dict['label'])
    ax.set_xlabel('Preferred depth (cm)', fontsize = fontsize_dict['label'])
    plt.xticks(fontsize=fontsize_dict['tick'])
    yticks = plt.gca().get_yticks()
    plt.yticks(yticks[0::2], fontsize=fontsize_dict['tick'])
    plotting_utils.despine()
    
    
def plot_psth_raster(
    fig,
    results_df,
    depth_list,
    use_cols=["preferred_depth_crossval", "psth_crossval", "preferred_depth_rsq"],
    depth_rsq_thr = 0.04,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    # Filter neurons with a depth tuning fit rsq threshold
    results_df = results_df[(results_df[use_cols[2]]>depth_rsq_thr) & (results_df["iscell"]==1)]
    psths = np.stack(results_df[use_cols[1]])[:,:-1,:] # exclude blank
    ndepths = psths.shape[1]-1
    nbins = psths.shape[2]
    
    # Sort neurons by preferred depth
    preferred_depths = results_df[use_cols[0]].values
    order = preferred_depths.argsort()
    psths = psths[order]
    psths = psths.reshape(psths.shape[0], -1)
    
    # Normalize PSTHs
    neuron_max = np.nanmax(psths, axis=1)[:, np.newaxis]
    neuron_min = np.nanmin(psths, axis=1)[:, np.newaxis]
    normed_psth = (psths - neuron_min) / (neuron_max-neuron_min)
    
    # Plot PSTHs
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    ax.imshow(normed_psth, vmax=1, aspect='auto', cmap=plotting_utils.generate_cmap(cmap_name="WhRd"))
    
    # Plot vertical lines to separate different depths
    for i in range(ndepths):
        ax.axvline((i+1)*nbins, color='k', linewidth=0.5)
        
    # Change xticks positions to the middle of current ticks and show depth at the tick position
    xticks = np.arange(0+0.5,ndepths+1+0.5)*nbins
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(depth_list))
    ax.set_xlabel('Preferred depth (cm)', fontsize=fontsize_dict['label'])
    ax.tick_params(axis='x', labelsize=fontsize_dict['tick'], rotation=60)
    ax.set_ylabel('Neuron no.', fontsize=fontsize_dict['label'])
    ax.tick_params(axis='y', labelsize=fontsize_dict['tick'])
    
    
def plot_depth_neuron_perc_hist(
    fig,
    results_df,
    use_col=["depth_tuning_test_spearmanr_pval_closedloop"],
    bins=50,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    results_df = results_df[results_df['iscell'] ==1]
    neuron_sum = results_df.groupby('session')[['roi']].agg(['count']).values.flatten()
    prop = results_df.groupby('session').apply(lambda x: x[x[use_col] < 0.05][['roi']].agg(['count'])).values.flatten()
    
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    ax.hist(prop/neuron_sum, bins=bins, color='k')
    xlim = ax.get_xlim()
    ax.set_xlim([0, xlim[1]])
    ax.set_xlabel('Proportion of depth-tuned neurons', fontsize=fontsize_dict['label'])
    ax.set_ylabel('Session number', fontsize=fontsize_dict['label'])
    plotting_utils.despine()