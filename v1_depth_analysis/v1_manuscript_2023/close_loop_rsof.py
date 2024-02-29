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
import seaborn as sns

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import spheres, find_depth_neurons, common_utils, fit_gaussian_blob, size_control
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
        linecolor = basic_vis_plots.get_depth_color(depth, depth_list, cmap=cm.cool.reversed())
        ax.plot(
            bin_centers[idepth, :],
            speed_tuning[idepth, :],
            color=linecolor,
            label=f"{int(depth_list[idepth] * 100)} cm",
            linewidth=linewidth,
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
            ax.set_xlabel("Optic flow speed "+u'(\N{DEGREE SIGN}/s)', fontsize=fontsize_dict["label"])

    # Plot tuning to gray period
    if which_speed == "RS":
        ax.plot(
            bin_centers[-1, :],
            speed_tuning[-1, :],
            color="gray",
            label=f"blank",
            linewidth=linewidth,
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
        ax.set_xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        
    if legend_on:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=fontsize_dict["legend"], frameon=False, handlelength=1)
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
    is_closed_loop=1,
    vmin=None,
    vmax=None,
    xlabel="Running speed (cm/s)",
    ylabel="Optical flow speed \n(degrees/s)",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    trials_df = trials_df[trials_df.closed_loop == is_closed_loop]
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
    
    if vmin is None:
        vmin = np.nanmax([0, np.percentile(bin_means[1:, 1:].flatten(), 1)])
    if vmax is None:
        vmax = np.nanmax(bin_means[1:, 1:].flatten())
    ax = fig.add_axes([plot_x, plot_y, plot_width*0.9, plot_height*0.9]) 
    im = ax.imshow(
        bin_means[1:,1:].T,
        origin="lower",
        aspect="equal",
        # cmap=generate_cmap(cmap_name="WhRd"),
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )

    ticks_select1, ticks_select2, bin_edges1, bin_edges2 = basic_vis_plots.get_RS_OF_heatmap_axis_ticks(
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
    
    if is_closed_loop:
        ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
        ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    if not is_closed_loop:
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        ax_left = fig.add_axes([plot_x+0.15*plot_width, 
                                plot_y, 
                                plot_width*0.9/(log_range["of_bin_num"]-1), 
                                plot_height*0.9])
        ax_left.imshow(
            bin_means[0,1:].reshape(1,-1).T,
            origin="lower",
            aspect="equal",
            # cmap=generate_cmap(cmap_name="WhRd"),
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            
        )
        plt.yticks(ticks_select2, bin_edges2)
        plt.xticks(ax_left.get_xticks(), "")
        
        ax_down = fig.add_axes([plot_x, 
                                plot_y-plot_height*0.9/(log_range["of_bin_num"]-1)*1.5, 
                                plot_width*0.9, 
                                plot_height*0.9/(log_range["of_bin_num"]-1)])
        ax_down.imshow(
            bin_means[1:,0].reshape(-1,1).T,
            origin="lower",
            aspect="equal",
            # cmap=generate_cmap(cmap_name="WhRd"),
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
            
        )
        plt.xticks(
            ticks_select1,
            bin_edges1,
            rotation=45,
            ha="center",
            fontsize=fontsize_dict["tick"],
        )
        plt.yticks(ax_down.get_yticks(), "")
        
        ax_corner = fig.add_axes([plot_x+0.15*plot_width,
                                  plot_y-plot_height*0.9/(log_range["of_bin_num"]-1)*1.5,
                                  plot_height*0.9/(log_range["of_bin_num"]-1),
                                  plot_height*0.9/(log_range["of_bin_num"]-1)])
        ax_corner.imshow(
            bin_means[0,0].reshape(1,1),
            origin="lower",
            aspect="equal",
            cmap="Reds",
            vmin=np.nanmax([0, np.percentile(bin_means[1:, 1:].flatten(), 1)]),
            vmax=np.nanmax(bin_means[1:,1:].flatten()),
        )
        plt.yticks(ax_corner.get_yticks()[1::2], ["< 0.03"])
        plt.xticks(ax_corner.get_xticks()[1::2], ["< 1"])
    
        ax_down.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
        ax_left.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
        ax_left.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax_down.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax_corner.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        
    ax2 = fig.add_axes([plot_x + plot_width*0.75, plot_y, cbar_width, plot_height*0.9]) 
    fig.colorbar(im, cax=ax2, label="\u0394F/F")
    ax2.tick_params(labelsize=fontsize_dict["legend"])
    ax2.set_ylabel("\u0394F/F", rotation=270, fontsize=fontsize_dict["legend"])

    extended_matrix = bin_means
    return extended_matrix


def plot_RS_OF_fitted_tuning(
    fig,
    neurons_df,
    roi,
    model="gaussian_2d",
    min_sigma=0.25,
    vmin=0,
    vmax=None,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    xlabel="Running speed (cm/s)",
    ylabel="Optical flow speed \n(degrees/s)",
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):

    """
    Plot the fitted tuning of a neuron.
    """
    rs = (
        np.logspace(
            log_range["rs_bin_log_min"], log_range["rs_bin_log_max"], 100, base=10
        )
        / 100
    )  # cm/s --> m/s
    of = np.logspace(
        log_range["of_bin_log_min"], log_range["of_bin_log_max"], 100, base=10
    )  # deg/s

    rs_grid, of_grid = np.meshgrid(np.log(rs), np.log(of))
    if model == "gaussian_2d":
        resp_pred = fit_gaussian_blob.gaussian_2d(
            (rs_grid, of_grid),
            *neurons_df["rsof_popt_closedloop_g2d"].iloc[roi],
            min_sigma=0.25,
        )
    elif model == "gaussian_additive":
        resp_pred = fit_gaussian_blob.gaussian_additive(
            (rs_grid, of_grid),
            *neurons_df["rsof_popt_closedloop_gadd"].iloc[roi],
            min_sigma=0.25,
        )
    elif model == "gaussian_OF":
        resp_pred = fit_gaussian_blob.gaussian_1d(
            of_grid, *neurons_df["rsof_popt_closedloop_gof"].iloc[roi], min_sigma=0.25
        )
    resp_pred = resp_pred.reshape((len(of), len(rs)))

    ax = fig.add_axes([plot_x, plot_y, plot_width*0.9, plot_height*0.9]) 
    im = ax.imshow(
        resp_pred,
        origin="lower",
        extent=[rs.min() * 100,rs.max() * 100, of.min(), of.max()],
        aspect=rs.max()
        * 100
        / of.max()
        * log_range["of_bin_num"]
        / log_range["rs_bin_num"],
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )
    ticks_select1 = ax.get_xticks()[:-1]
    ticks_select2 = ax.get_yticks()[:-1]
    plt.xticks(
        ticks_select1,
        np.round(np.geomspace(rs.min() * 100, rs.max() * 100, len(ticks_select1))),
        rotation=45,
        ha="center",
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(ticks_select2, 
               np.round(np.geomspace(of.min(), of.max(), len(ticks_select2))), 
               fontsize=fontsize_dict["tick"])
    
    ax2 = fig.add_axes([plot_x + plot_width*0.75, plot_y, cbar_width, plot_height*0.9]) 
    fig.colorbar(im, cax=ax2, label="\u0394F/F")
    ax2.tick_params(labelsize=fontsize_dict["legend"])
    ax2.set_ylabel("\u0394F/F", rotation=270, fontsize=fontsize_dict["legend"])
    
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    return resp_pred.min(), resp_pred.max()


def plot_r2_comparison(
    fig,
    neurons_df,
    use_cols,
    labels,
    plot_type="violin",
    markersize=10,
    alpha=0.3,
    color='k',
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    if plot_type == "violin":
        results = pd.DataFrame(columns=["model","rsq"])
        neurons_df = neurons_df[
            (neurons_df["iscell"] == 1) & 
            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05) 
            ]
        ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
        for i, col in enumerate(use_cols):
            neurons_df[col][neurons_df[col]<-1] = 0
            results = pd.concat([results, pd.DataFrame({"model": labels[i], 
                                                        "rsq": neurons_df[col]}, )
                                ],
                                ignore_index=True)
        sns.violinplot(data=results, x="model", y="rsq", ax=ax)
        ax.set_ylabel("R-squared", fontsize=fontsize_dict["label"])
        ax.set_xlabel("Model", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        plotting_utils.despine()
        
        print(f"{labels[0]} vs {labels[1]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[0]], results['rsq'][results['model'] == labels[1]])}")
        print(f"{labels[0]} vs {labels[2]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[0]], results['rsq'][results['model'] == labels[2]])}")
    
    elif plot_type == "bar":
        # Find the best model for each neuron
        neurons_df["best_model"] = neurons_df[use_cols].idxmax(axis=1)
        name_mapping = {
            use_cols[0]: "g2d",
            use_cols[1]: "gadd",
            use_cols[2]: "gof",
        }
        neurons_df["best_model"] = neurons_df["best_model"].map(name_mapping)
        
        # Filter depth neurons
        neurons_df_filtered = neurons_df[
            (neurons_df["iscell"] == 1) & 
            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05) 
            ]
        
        ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
        # Calculate percentage of neurons that have the best model
        neuron_sum = neurons_df_filtered.groupby('session')[['roi']].agg(['count']).values.flatten()
        props = []
        for i, model in enumerate(["g2d", "gadd", "gof"]):
            prop = (neurons_df_filtered.groupby('session').apply(lambda x: x[x["best_model"]==model][['roi']].agg(['count'])).values.flatten())/neuron_sum
            props.append(prop)
            # Plot bar plot 
            ax.bar(x=0.5*i, 
                height=prop.mean(), 
                width=0.2,
                yerr=scipy.stats.sem(prop),
                capsize=10,
                color=color, 
                alpha=0.5,)
        
            ax.scatter(x=np.ones(len(prop))*0.5*i, 
                    y=prop, 
                    color=color, 
                    s=markersize,
                    alpha=alpha)

        ax.set_xticks([0,0.5,1])
        ax.set_xticklabels(labels, fontsize=fontsize_dict["label"])
        ax.set_ylabel('Proportion of depth neurons\' \nbest model', fontsize=fontsize_dict['label'])
        ax.set_ylim([0,1])
        plotting_utils.despine()
        ax.tick_params(axis='y', which='major', labelsize=fontsize_dict['tick'])
        print(f"{labels[0]} vs {labels[1]}: {scipy.stats.wilcoxon(props[0],props[1])}")
        print(f"{labels[0]} vs {labels[2]}: {scipy.stats.wilcoxon(props[0],props[2])}")

            

def plot_speed_depth_scatter(
    fig,
    neurons_df,
    xcol,
    ycol,
    xlabel="Running speed (cm/s)",
    ylabel="Preferred depth (cm)",
    s=10,
    alpha=0.2,
    c='g',
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    aspect_equal=False,
    plot_diagonal=False,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},):
    
    # Filter neurons_df
    neurons_df = neurons_df[(neurons_df["iscell"] == 1) & 
                            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05) &
                            (neurons_df["rsof_rsq_closedloop_g2d"] > 0.02)
                            ]
    
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
    X = neurons_df[xcol].values
    y = neurons_df[ycol].values
    ax.scatter(X, y, s=s, alpha=alpha, c=c,  edgecolors="none")
    if plot_diagonal:
        ax.plot(np.geomspace(y.min(),y.max(),1000), np.geomspace(y.min(),y.max(),1000), c='y', linestyle="dotted", linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    if aspect_equal:
        ax.set_aspect("equal")
    plotting_utils.despine()
    r, p = scipy.stats.spearmanr(X, y)
    ax.set_title(f"R = {r:.2f}, p = {p:.2e}", fontsize=fontsize_dict["title"])
    print(f"Correlation between {xcol} and {ycol}: {scipy.stats.spearmanr(X, y)}")


def plot_speed_colored_by_depth(
    fig,
    neurons_df,
    xcol,
    ycol,
    zcol,
    xlabel="Running speed (cm/s)",
    ylabel="Optic flow speed (degree/s)",
    zlabel="Preferred depth (cm)",
    s=10,
    alpha=0.2,
    cmap='cool_r',
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
    ):
    
    # Filter neurons_df
    neurons_df = neurons_df[(neurons_df["iscell"] == 1) & 
                            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05) &
                            (neurons_df["rsof_rsq_closedloop_g2d"] > 0.02)
                            ]
    
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
    sns.scatterplot(
        neurons_df, 
        x=xcol, 
        y=ycol, 
        hue=np.log(neurons_df[zcol]),
        # hue_norm = (np.log(6), np.log(600)),
        palette="cool_r",
        s=s, 
        alpha=alpha,
        ax=ax,
        linewidth=0,)
    sns.despine()
    ax.set_aspect("equal", "box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.get_legend().remove()
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    
    norm = matplotlib.colors.LogNorm(np.nanmin(neurons_df[zcol]), np.nanmax(neurons_df[zcol]))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Remove the legend and add a colorbar
    ax2 = fig.add_axes([plot_x + plot_width + cbar_width, plot_y+0.05, cbar_width, plot_height*0.88]) 
    ax2.figure.colorbar(sm, cax=ax2)
    ax2.set_ylabel(zlabel, rotation=270, fontsize=fontsize_dict['label'])
    ax2.tick_params(labelsize=fontsize_dict['legend'])
    ax2.get_yaxis().labelpad = 15

    # cbar = ax.figure.colorbar(sm)
    # cbar.ax.set_ylabel(zlabel, rotation=270, fontsize=fontsize_dict['legend'])
    # cbar.ax.tick_params(labelsize=fontsize_dict['legend'])
    # cbar.ax.get_yaxis().labelpad = 25
    # yticks = cbar.ax.get_yticks()
