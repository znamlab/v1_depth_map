"""
Helper functions to plot RSOF integration figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from cottage_analysis.plotting import rsof_plots, depth_selectivity_plots


def plot_example_neuron_rsof(
    fig,
    roi,
    iroi,
    neurons_df_example,
    trials_df_example,
    models,
    model_labels,
    fontsize_dict,
):
    """
    Plots RS/OF tuning curves and fits for an example ROI.
    """
    fig.add_axes([0.06, 0.88 - 0.43 * iroi, 0.15, 0.1])
    depth_tuning_kwargs = dict(
        rs_thr=None,
        plot_fit=True,
        plot_smooth=False,
        linewidth=1.5,
        linecolor="royalblue",
        closed_loop=1,
        fontsize_dict=fontsize_dict,
        markersize=8,
        markeredgecolor="w",
    )
    depth_selectivity_plots.plot_depth_tuning_curve(
        neurons_df=neurons_df_example,
        trials_df=trials_df_example,
        roi=roi,
        **depth_tuning_kwargs,
    )
    fig.add_axes([0.25, 0.88 - 0.43 * iroi, 0.15, 0.1])
    rsof_plots.plot_speed_tuning(
        trials_df=trials_df_example,
        roi=roi,
        is_closed_loop=1,
        nbins=15,
        which_speed="RS",
        speed_min=0,
        speed_max=1.5,
        speed_thr=0,
        smoothing_sd=1,
        markersize=3,
        linewidth=1,
        markeredgecolor="w",
        fontsize_dict=fontsize_dict,
        legend_on=False,
    )
    ylim = plt.gca().get_ylim()

    fig.add_axes([0.44, 0.88 - 0.43 * iroi, 0.15, 0.1])
    rsof_plots.plot_speed_tuning(
        trials_df=trials_df_example,
        roi=roi,
        is_closed_loop=1,
        nbins=20,
        which_speed="OF",
        speed_min=0.01,
        speed_max=1.5,
        speed_thr=0.01,
        of_min=1e-2,
        of_max=1e4,
        smoothing_sd=1,
        markersize=3,
        linewidth=1,
        markeredgecolor="w",
        fontsize_dict=fontsize_dict,
        legend_on=True,
        ylim=ylim,
    )
    plt.gca().set_ylabel("")

    fig.add_axes([0.04, 0.67 - 0.43 * iroi, 0.13, 0.13])
    vmin, vmax = rsof_plots.plot_RS_OF_matrix(
        trials_df=trials_df_example,
        roi=roi,
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
        vmin=0,
        vmax=1,
        xlabel="Running speed (cm/s)",
        ylabel="Optic flow speed (degrees/s)",
        cbar_width=0.01,
        fontsize_dict=fontsize_dict,
    )

    # 1 example neuron, fits of 5 models
    for imodel, (model, model_label) in enumerate(zip(models, model_labels)):
        if imodel == 0:
            ylabel = "Optic flow speed (degrees/s)"
        else:
            ylabel = ""
        if imodel == 1:
            xlabel = "Running speed (cm/s)"
        else:
            xlabel = ""

        fig.add_axes([0.24 + 0.08 * imodel, 0.67 - 0.43 * iroi, 0.1, 0.1])
        rsof_plots.plot_RS_OF_fit(
            neurons_df=neurons_df_example,
            roi=roi,
            model=model,
            model_label=model_label,
            min_sigma=0.25,
            vmin=0,
            vmax=1,
            log_range={
                "rs_bin_log_min": 0,
                "rs_bin_log_max": 2.5,
                "rs_bin_num": 6,
                "of_bin_log_min": -1.5,
                "of_bin_log_max": 3.5,
                "of_bin_num": 11,
                "log_base": 10,
            },
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize_dict=fontsize_dict,
            cbar_width=None,
        )
        if imodel > 0:
            plt.gca().set_yticks([])


def plot_expected_depth_vs_treadmill(
    ax, df, fontsize_dict, ticks=(5, 10, 20, 40, 80, 160, 320, 640), **kwargs
):
    """
    Plots the expected closed-loop depth (from preferred RS and OF ratio)
    vs preferred depth with treadmill.
    """

    # Data is natively in cm (RS: cm/s, OF: rad/s => RS/OF: cm)
    expected_depth = (
        df["preferred_RS_closedloop_crossval_g2d_treadmill"]
        / df["preferred_OF_closedloop_crossval_g2d_treadmill"]
    )
    treadmill_depth = df["preferred_depth_closedloop_crossval"]

    ax.plot(np.log([2, 1000]), np.log([2, 1000]), "--", color="grey")
    sc = ax.scatter(np.log(expected_depth), np.log(treadmill_depth), **kwargs)
    ax.set_aspect("equal")
    ax.set_ylabel(
        "Preferred depth\nin closed loop (cm)", fontsize=fontsize_dict["label"]
    )
    ax.set_xlabel(
        "Ratio of preferred RS and OF\nwith treadmill (cm)",
        fontsize=fontsize_dict["label"],
    )
    # Place ticks at the presented depth values
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(ticks)
    ax.set_yticks(np.log(ticks))
    ax.set_yticklabels(ticks)
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])

    return sc


def plot_treadmill_vs_closedloop_comparison(
    fig,
    trials_df_tread,
    trials_df_sphere,
    roi,
    fontsize_dict,
    plot_x=0.35,
    plot_y=0.1,
    plot_width=0.25,
    plot_height=0.15,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
    **kwargs
):
    """
    Plots side-by-side matrices for Closed-loop and Treadmill.
    """
    ax_sphere = fig.add_axes([plot_x, plot_y, plot_width / 2.2, plot_height])
    ax_tread = fig.add_axes(
        [plot_x + plot_width / 1.8, plot_y, plot_width / 2.2, plot_height]
    )

    rsof_plots.plot_RS_OF_matrix(
        trials_df_sphere,
        roi,
        log_range=log_range,
        is_closed_loop=1,
        title="Closed-loop",
        ax=ax_sphere,
        fontsize_dict=fontsize_dict,
        cbar_width=None,
        **kwargs,
    )

    rsof_plots.plot_RS_OF_matrix(
        trials_df_tread,
        roi,
        log_range=log_range,
        is_closed_loop=1,
        title="Treadmill",
        ax=ax_tread,
        fontsize_dict=fontsize_dict,
        **kwargs,
    )
    ax_tread.set_ylabel("")
    ax_tread.set_yticklabels([])

    return [ax_sphere, ax_tread]


def plot_treadmill_diagonal_and_traces(
    fig,
    trials_df_tread,
    roi,
    fontsize_dict,
    plot_x=0.06,
    plot_y=0.1,
    matrix_size=0.15,
    trace_width=0.4,
    log_range={
        "rs_bin_log_min": 0,
        "rs_bin_log_max": 2.5,
        "rs_bin_num": 6,
        "of_bin_log_min": -1.5,
        "of_bin_log_max": 3.5,
        "of_bin_num": 11,
        "log_base": 10,
    },
):
    """
    Plots a treadmill matrix with a diagonal line (pref RS = pref OF)
    and placeholder for traces.
    """
    ax_mat = fig.add_axes([plot_x, plot_y, matrix_size, matrix_size])
    rsof_plots.plot_RS_OF_matrix(
        trials_df=trials_df_tread,
        roi=roi,
        log_range=log_range,
        ax=ax_mat,
        xlabel="Running speed (cm/s)",
        ylabel="Optic flow speed (degrees/s)",
        fontsize_dict=fontsize_dict,
        cbar_width=None,
    )
    ax_mat.plot(
        np.log10([1, 640]), np.log10([1, 640]), "--", color="grey", linewidth=1.5
    )

    ax_trace = fig.add_axes(
        [plot_x + matrix_size + 0.08, plot_y, trace_width, matrix_size]
    )
    ax_trace.set_title("DFF traces for treadmill", fontsize=fontsize_dict["title"])
    ax_trace.set_ylabel("DFF", fontsize=fontsize_dict["label"])
    ax_trace.set_xlabel("Time (s)", fontsize=fontsize_dict["label"])
    ax_trace.tick_params(labelsize=fontsize_dict["tick"])


def plot_gaussian_theta_distribution(
    ax, neurons_df, fontsize_dict, col="rsof_popt_closedloop_g2d_treadmill"
):
    """
    Plots a histogram of the 'theta' parameter from 2D Gaussian fits on treadmill data.
    """
    thetas = []
    for popt in neurons_df[col].values:
        if isinstance(popt, (list, np.ndarray)) and len(popt) >= 6:
            thetas.append(popt[5])

    if len(thetas) == 0:
        ax.text(0.5, 0.5, "No theta data", ha="center")
        return

    thetas_deg = np.degrees(thetas)

    ax.hist(thetas_deg, bins=30, color="orange", alpha=0.7, edgecolor="k")
    ax.set_xlabel("Gaussian fit angle (degrees)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Number of neurons", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])

    return ax
