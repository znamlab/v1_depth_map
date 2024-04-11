import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import cm

import scipy
import seaborn as sns

from cottage_analysis.analysis import (
    find_depth_neurons,
    fit_gaussian_blob,
)
from cottage_analysis.plotting import basic_vis_plots, plotting_utils
from cottage_analysis.analysis import common_utils


def calculate_speed_tuning(speed_arr, dff_arr, bins, smoothing_sd=1, ci_range=0.95):
    # calculate speed tuning
    bin_means, _, _ = scipy.stats.binned_statistic(
        x=speed_arr,
        values=dff_arr,
        statistic="mean",
        bins=bins,
    )
    bin_counts, _, _ = scipy.stats.binned_statistic(
        x=speed_arr,
        values=dff_arr,
        statistic="count",
        bins=bins,
    )
    ci = np.zeros((len(bin_means), 2)) * np.nan
    for ibin in range(len(bin_means)):
        idx = (speed_arr > bins[ibin]) & (speed_arr < bins[ibin + 1])
        if np.sum(idx) > 0:
            ci[ibin, 0], ci[ibin, 1] = common_utils.get_bootstrap_ci(
                dff_arr[idx], n_bootstraps=1000, sig_level=1 - ci_range
            )
    smoothed_tuning = plotting_utils.get_tuning_function(
        bin_means, bin_counts, smoothing_sd=smoothing_sd
    )
    return bin_means, smoothed_tuning, ci


def plot_speed_tuning(
    fig,
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
    ci_range=0.95,
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
        depth_list.append("blank")
        bins = (
            np.linspace(start=speed_min, stop=speed_max, num=nbins + 1, endpoint=True)
            * 100
        )
    tuning = np.zeros(((len(depth_list)), nbins))
    smoothed_tuning = np.zeros(((len(depth_list)), nbins))
    ci = np.zeros(((len(depth_list)), nbins, 2))
    bin_centers = np.zeros(((len(depth_list)), nbins))

    # Find all speed and dff of this ROI for a specific depth
    for idepth, depth in enumerate(depth_list):
        if depth == "blank":
            all_speed = trials_df[f"{which_speed}_blank"].values
            all_dff = trials_df["dff_blank"].values
        else:
            all_speed = grouped_trials.get_group(depth)[f"{which_speed}_stim"].values
            all_dff = grouped_trials.get_group(depth)["dff_stim"].values
        dff_arr = np.array([j for i in all_dff for j in i[:, roi]])
        speed_arr = np.array([j for i in all_speed for j in i])
        if which_speed == "OF":
            speed_arr = np.degrees(speed_arr)  # rad --> degrees
        else:
            speed_arr = speed_arr * 100  # m/s --> cm/s
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
        tuning[idepth], smoothed_tuning[idepth], ci[idepth] = calculate_speed_tuning(
            speed_arr,
            dff_arr,
            bins,
            smoothing_sd=smoothing_sd,
            ci_range=ci_range,
        )
    # Plotting
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    for idepth, depth in enumerate(depth_list):
        if depth == "blank":
            linecolor = "gray"
            label = "blank"
        else:
            linecolor = basic_vis_plots.get_depth_color(
                depth, depth_list, cmap=cm.cool.reversed()
            )
            label = f"{int(depth_list[idepth] * 100)} cm"
        ax.plot(
            bin_centers[idepth, :],
            smoothed_tuning[idepth, :],
            color=linecolor,
            label=label,
            linewidth=linewidth,
        )
        ax.errorbar(
            x=bin_centers[idepth, :],
            y=tuning[idepth, :],
            yerr=np.abs(ci[idepth, :].T - tuning[idepth, :]),
            fmt="o",
            color=linecolor,
            ls="none",
            markersize=markersize,
            linewidth=linewidth,
        )
        if which_speed == "OF":
            ax.set_xscale("log")
    # Plot tuning to gray period
    if which_speed == "RS":
        ax.set_xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
    else:
        ax.set_xlabel(
            "Optic flow speed (degrees/s)",
            fontsize=fontsize_dict["label"],
        )
    ax.set_ylabel("\u0394F/F", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])

    if legend_on:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.95, 1.15),
            fontsize=fontsize_dict["legend"],
            frameon=False,
            handlelength=1,
            labelspacing=0.35,
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
    ax = fig.add_axes([plot_x, plot_y, plot_width * 0.9, plot_height * 0.9])
    im = ax.imshow(
        bin_means[1:, 1:].T,
        origin="lower",
        aspect="equal",
        # cmap=generate_cmap(cmap_name="WhRd"),
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )

    ticks_select1, ticks_select2, bin_edges1, bin_edges2 = (
        basic_vis_plots.get_RS_OF_heatmap_axis_ticks(
            log_range=log_range, fontsize_dict=fontsize_dict
        )
    )
    plt.xticks(
        ticks_select1[0::2],
        bin_edges1[0::2],
        fontsize=fontsize_dict["tick"],
    )

    plt.yticks(ticks_select2[1::2], bin_edges2[1::2], fontsize=fontsize_dict["tick"])

    if is_closed_loop:
        ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
        ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    if not is_closed_loop:
        ax.set_xticks([])
        ax.set_yticks([])
        ax_left = fig.add_axes(
            [
                plot_x + 0.15 * plot_width,
                plot_y,
                plot_width * 0.9 / (log_range["of_bin_num"] - 1),
                plot_height * 0.9,
            ]
        )
        ax_left.imshow(
            bin_means[0, 1:].reshape(1, -1).T,
            origin="lower",
            aspect="equal",
            # cmap=generate_cmap(cmap_name="WhRd"),
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        plt.yticks(ticks_select2[1::2], bin_edges2[1::2], fontsize=fontsize_dict["tick"])
        plt.xticks([])

        ax_down = fig.add_axes(
            [
                plot_x,
                plot_y - plot_height * 0.9 / (log_range["of_bin_num"] - 1) * 1.5,
                plot_width * 0.9,
                plot_height * 0.9 / (log_range["of_bin_num"] - 1),
            ]
        )
        ax_down.imshow(
            bin_means[1:, 0].reshape(-1, 1).T,
            origin="lower",
            aspect="equal",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        plt.xticks(
            ticks_select1[0::2],
            bin_edges1[0::2],
            fontsize=fontsize_dict["tick"]
        )
        plt.yticks([])

        ax_corner = fig.add_axes(
            [
                plot_x + 0.15 * plot_width,
                plot_y - plot_height * 0.9 / (log_range["of_bin_num"] - 1) * 1.5,
                plot_height * 0.9 / (log_range["of_bin_num"] - 1),
                plot_height * 0.9 / (log_range["of_bin_num"] - 1),
            ]
        )
        ax_corner.imshow(
            bin_means[0, 0].reshape(1, 1),
            origin="lower",
            aspect="equal",
            cmap="Reds",
            vmin=vmin,
            vmax=vmax,
        )
        plt.yticks(ax_corner.get_yticks()[1::2], ["< 0.03"])
        plt.xticks(ax_corner.get_xticks()[1::2], ["< 1"])

        ax_down.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
        ax_left.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
        ax_left.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax_down.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        ax_corner.tick_params(
            axis="both", which="major", labelsize=fontsize_dict["tick"]
        )

    ax2 = fig.add_axes(
        [plot_x + plot_width * 0.75, plot_y, cbar_width, plot_height * 0.9]
    )
    fig.colorbar(im, cax=ax2, label="\u0394F/F")
    ax2.tick_params(labelsize=fontsize_dict["legend"])
    ax2.set_ylabel("\u0394F/F", rotation=270, fontsize=fontsize_dict["legend"])

    return vmin, vmax


def plot_RS_OF_fit(
    fig,
    neurons_df,
    roi,
    model="gaussian_2d",
    model_label="",
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
    if model == "gof":
        params = of_grid
    elif model == "gratio":
        params = rs_grid - of_grid
    else:
        params = (rs_grid, of_grid)
    funcs = {
        "g2d": fit_gaussian_blob.gaussian_2d,
        "gadd": fit_gaussian_blob.gaussian_additive,
        "gof": fit_gaussian_blob.gaussian_1d,
        "gratio": fit_gaussian_blob.gaussian_1d,
    }
    resp_pred = funcs[model](
        params,
        *neurons_df[f"rsof_popt_closedloop_{model}"].iloc[roi],
        min_sigma=min_sigma,
    ).reshape((len(of), len(rs)))

    ax = fig.add_axes([plot_x, plot_y, plot_width * 0.9, plot_height * 0.9])
    im = ax.imshow(
        resp_pred,
        origin="lower",
        extent=[
            log_range["rs_bin_log_min"],
            log_range["rs_bin_log_max"],
            log_range["of_bin_log_min"],
            log_range["of_bin_log_max"],
        ],
        aspect="equal",
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xticks(
        [0, 1, 2],
        labels=["1", "10", "100"],
        fontsize=fontsize_dict["tick"],
    )
    plt.yticks(
        [-1, 0, 1, 2, 3],
        labels=["0.1", "1", "10", "100", "1000"],
        fontsize=fontsize_dict["tick"],
    )
    if cbar_width is not None:
        ax2 = fig.add_axes(
            [plot_x + plot_width * 0.75, plot_y, cbar_width, plot_height * 0.9]
        )
        fig.colorbar(im, cax=ax2, label="\u0394F/F")
        ax2.tick_params(labelsize=fontsize_dict["legend"])
        ax2.set_ylabel("\u0394F/F", rotation=270, fontsize=fontsize_dict["legend"])
    plt.title(
        model_label,
        fontdict={"fontsize": fontsize_dict["label"]},
    )
    plt.text(
        x=log_range["rs_bin_log_min"] + 0.2,
        y=log_range["of_bin_log_max"] - 0.7,
        s=f"$R^2$ = {neurons_df[f'rsof_test_rsq_closedloop_{model}'].iloc[roi]:.2f}",
        fontsize=fontsize_dict["tick"],
    )
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    return resp_pred.min(), resp_pred.max()


def plot_r2_comparison(
    fig,
    neurons_df,
    models,
    labels,
    plot_type="violin",
    markersize=10,
    alpha=0.3,
    color="k",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    if plot_type == "violin":
        results = pd.DataFrame(columns=["model", "rsq"])
        ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
        for i, col in enumerate(models):
            neurons_df[col][neurons_df[col] < -1] = 0
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        {"model": labels[i], "rsq": neurons_df[col]},
                    ),
                ],
                ignore_index=True,
            )
        sns.violinplot(data=results, x="model", y="rsq", ax=ax)
        ax.set_ylabel("R-squared", fontsize=fontsize_dict["label"])
        ax.set_xlabel("Model", fontsize=fontsize_dict["label"])
        ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
        plotting_utils.despine()

        print(
            f"{labels[0]} vs {labels[1]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[0]], results['rsq'][results['model'] == labels[1]])}"
        )
        print(
            f"{labels[0]} vs {labels[2]}: {scipy.stats.wilcoxon(results['rsq'][results['model'] == labels[0]], results['rsq'][results['model'] == labels[2]])}"
        )

    elif plot_type == "bar":
        model_cols = [f"rsof_test_rsq_closedloop_{model}" for model in models]
        # Find the best model for each neuron
        neurons_df["best_model"] = neurons_df[model_cols].idxmax(axis=1)

        ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
        # Calculate percentage of neurons that have the best model
        neuron_sum = (
            neurons_df.groupby("session")[["roi"]].agg(["count"]).values.flatten()
        )
        props = []
        # calculate the proportion of neurons that have the best model for each session
        for i, model in enumerate(model_cols):
            prop = (
                neurons_df.groupby("session")
                .apply(lambda x: x[x["best_model"] == model][["roi"]].agg(["count"]))
                .values.flatten()
            ) / neuron_sum
            props.append(prop)
            # Plot bar plot
            sns.stripplot(
                x=np.ones(len(prop)) * i,
                y=prop,
                size=markersize,
                alpha=alpha,
                jitter=0.4,
                edgecolor="white",
                color=sns.color_palette("Set1")[i],
            )
            plt.plot(
                [i - 0.4, i + 0.4],
                [np.median(prop), np.median(prop)],
                linewidth=3,
                color=color,
            )
        sns.despine(offset=5, ax=plt.gca())
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(labels, fontsize=fontsize_dict["label"], rotation=90)
        ax.set_ylabel(
            "Proportion of neurons \nwith best model fit",
            fontsize=fontsize_dict["label"],
        )
        ax.set_ylim([0, 1])
        ax.tick_params(axis="y", which="major", labelsize=fontsize_dict["tick"])
        print(f"{labels[0]} vs {labels[1]}: {scipy.stats.wilcoxon(props[0],props[1])}")
        print(f"{labels[0]} vs {labels[2]}: {scipy.stats.wilcoxon(props[0],props[2])}")


def plot_r2_cdfs(
    neurons_df,
    models,
    model_labels,
    xlim=(10**-4, 1),
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    def cdf(values):
        x = np.sort(values)
        y = np.linspace(0, 1, len(x) + 1)
        return x, y[1:]

    neurons_df_sig = neurons_df[
        (neurons_df["iscell"] == 1)
        & (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.001)
        & (neurons_df["preferred_depth_amplitude"] > 0.5)
    ]
    for model, label in zip(models, model_labels):
        plt.plot(*cdf(neurons_df_sig[f"rsof_test_rsq_closedloop_{model}"]), label=label)
    plt.xscale("log")
    plt.legend(frameon=False, fontsize=fontsize_dict["label"])
    plt.xlim(xlim)
    plt.ylim([0, 1])
    plt.gca().tick_params(axis="both", labelsize=fontsize_dict["tick"])
    plt.xlabel("$R^2$", fontsize=fontsize_dict["label"])
    plt.ylabel("Cumulative proportion of neurons", fontsize=fontsize_dict["label"])
    sns.despine(offset=5, ax=plt.gca())


def plot_r2_violin(
    neurons_df,
    models,
    model_labels,
    ylim=(10**-4, 1),
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):

    cols = [f"rsof_test_rsq_closedloop_{model}" for model in models]
    df = neurons_df[cols].melt(var_name="model", value_name="r2")
    df["model"] = df["model"].apply(lambda x: model_labels[cols.index(x)])
    df["r2"][df["r2"] < ylim[0]] = ylim[0]
    df["r2"][df["r2"] > ylim[1]] = ylim[1]
    sns.violinplot(
        data=df,
        y="r2",
        x="model",
        log_scale=True,
        hue="model",
        cut=0,
        inner="quartile",
        legend=False,
        fill=False,
        palette="Set1",
    )
    plt.ylim(ylim)
    plt.gca().tick_params(axis="y", labelsize=fontsize_dict["tick"])
    plt.gca().tick_params(axis="x", labelsize=fontsize_dict["label"], rotation=90)
    plt.xlabel("")
    # change the first xtick label
    ytick_labels = plt.gca().get_yticklabels()
    ytick_labels[1].set_text(f"\u2264 {ytick_labels[1].get_text()}")
    plt.gca().set_yticklabels(ytick_labels)
    plt.ylabel("$R^2$", fontsize=fontsize_dict["label"])
    sns.despine(offset=5, ax=plt.gca())


def plot_scatter(
    fig,
    neurons_df,
    xcol,
    ycol,
    xlabel="Running speed (cm/s)",
    ylabel="Preferred depth (cm)",
    s=10,
    alpha=0.2,
    c="g",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    aspect_equal=False,
    plot_diagonal=False,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    X = neurons_df[xcol].values
    y = neurons_df[ycol].values
    ax.scatter(X, y, s=s, alpha=alpha, c=c, edgecolors="none")
    if plot_diagonal:
        ax.plot(
            [y.min(), y.max()],
            [y.min(), y.max()],
            c="r",
            linestyle="--",
            linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    if aspect_equal:
        ax.set_aspect("equal")
    plotting_utils.despine()
    r, p = scipy.stats.spearmanr(X, y)
    print(f"Correlation between {xcol} and {ycol}: R = {r}, p = {p}")


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
    cmap="cool_r",
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
):
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
    )
    sns.despine()
    ax.set_aspect("equal", "box")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.get_legend().remove()
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])

    norm = matplotlib.colors.LogNorm(
        np.nanmin(neurons_df[zcol]), np.nanmax(neurons_df[zcol])
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # # Remove the legend and add a colorbar
    # ax2 = fig.add_axes(
    #     [
    #         plot_x + plot_width * 0.9,
    #         plot_y + 0.05,
    #         cbar_width,
    #         plot_height * 0.88,
    #     ]
    # )
    cbar = plt.colorbar(sm, shrink=0.5, ax=ax)
    cbar.ax.set_ylabel(
        zlabel, rotation=270, fontsize=fontsize_dict["label"], labelpad=10
    )
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"])

    # cbar = ax.figure.colorbar(sm)
    # cbar.ax.set_ylabel(zlabel, rotation=270, fontsize=fontsize_dict['legend'])
    # cbar.ax.tick_params(labelsize=fontsize_dict['legend'])
    # cbar.ax.get_yaxis().labelpad = 25
    # yticks = cbar.ax.get_yticks()
