"""
Helper functions to plot RSOF integration figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.patches import Ellipse, Rectangle
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
    Plots the expected treadmill depth (from preferred RS and OF ratio)
    vs preferred depth in closed loop.
    """
    ticks = np.asarray(ticks)
    # Data is natively in cm (RS: cm/s, OF: rad/s => RS/OF: cm)
    expected_depth = (
        df["preferred_RS_closedloop_crossval_g2d_treadmill"]
        / df["preferred_OF_closedloop_crossval_g2d_treadmill"]
    )
    treadmill_depth = df["preferred_depth_closedloop_crossval"]

    ax.plot(
        np.log([ticks.min(), ticks.max()]),
        np.log([ticks.min(), ticks.max()]),
        "--",
        color="grey",
    )
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


def add_ellipse_schematics(ax, plot_angle=True, plot_ecc=True, frame=True):
    """
    Add oriented, gradient-filled ellipses to a polar plot to visualize tuning markers and eccentricity scales.

    Args:
        ax (matplotlib.axes.PolarAxes): The polar axes to which the ellipses will be added.
        plot_angle (bool, optional): Whether to plot the orientation ellipses around the
            perimeter. Default is True.
        plot_ecc (bool, optional): Whether to plot the eccentricity legend ellipses along a
            radial axis. Default is True.
        frame (bool, optional): Whether to draw a white rectangle with a thin black border
            around each ellipse. Default is True.
    """
    fig = ax.get_figure()

    # PLOT ANGLE ELLIPSES
    # Fixed frame size (same for every angle ellipse), in axes-fraction units
    size = 0.1  # Roughly the size of the label text

    if plot_angle:
        ecc = 0.95
        ratio = np.sqrt(1 - ecc**2)

        r_pos = 1.22  # Just outside the r=1 limit
        for angle, theta_pos in zip([-45, 0, 45, 90, 135], [45, 90, -45, 0, 135]):
            angle = 0  # vertical orientation
            theta_pos = np.radians(theta_pos)
            # Blended transform: position in data (polar) coords, shape in axes-fraction units
            trans = ax.get_xaxis_transform()
            # Optional frame — drawn BEFORE gradient layers so it sits behind the ellipse
            if frame:
                point_to_pixel = transforms.Affine2D().scale(fig.dpi / 72.0)
                anchor = transforms.ScaledTranslation(
                    theta_pos, r_pos, ax.get_xaxis_transform()
                )
                frame_trans = point_to_pixel + anchor
                frame_half_pts = 10  # same as eccentricity frames → consistent look
                rect = Rectangle(
                    xy=(-frame_half_pts, -frame_half_pts),
                    width=frame_half_pts * 1.8,
                    height=frame_half_pts * 2,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.5,
                    transform=frame_trans,
                    clip_on=False,
                )
                ax.add_patch(rect)
            # Draw concentric ellipses for a "gradient" effect
            n_layers = 15
            for i in range(n_layers):
                alpha = (i + 1) / n_layers
                scale = 1 - (i / n_layers) * 0.8
                el = Ellipse(
                    xy=(theta_pos, r_pos),
                    width=size * scale * ratio,
                    height=size * scale,
                    angle=angle,
                    facecolor="red",
                    alpha=alpha * 0.3,
                    edgecolor="none",
                    transform=trans,
                    clip_on=False,
                )
                ax.add_patch(el)
            # Core ellipse for sharpness
            core_el = Ellipse(
                xy=(theta_pos, r_pos),
                width=size * 0.2 * ratio,
                height=size * 0.2,
                angle=angle,
                facecolor="red",
                alpha=0.5,
                edgecolor="none",
                transform=trans,
                clip_on=False,
            )
            ax.add_patch(core_el)
    if plot_ecc:
        # PLOT CIRCULARITY ELLIPSES
        eccs_leg = [0.2, 0.4, 0.6, 0.8]
        r_leg = [0.3, 0.46, 0.65, 0.82]
        theta_val = [-1.8, -1.35, -1.2, -1.1]
        size_pts = 10
        angle_leg = 45
        # Fixed square frame of side 2*frame_half_pts points (same for all eccentricities).
        # size_pts=10 is enough to enclose even the most eccentric ellipse (ecc=0.8)
        # rotated at 45°: its bounding half-size is only ~7 pts.
        frame_half_pts = size_pts
        for ecc, r_pos, theta in zip(eccs_leg, r_leg, theta_val):
            ratio = np.sqrt(1 - ecc**2)
            data_to_pixel = transforms.ScaledTranslation(theta, r_pos, ax.transData)
            point_to_pixel = transforms.Affine2D().scale(fig.dpi / 72.0)
            trans = point_to_pixel + data_to_pixel
            size_ = size_pts / ratio
            # Optional frame — drawn BEFORE gradient layers so it sits behind the ellipse
            if frame:
                rect = Rectangle(
                    xy=(-frame_half_pts, -frame_half_pts),
                    width=frame_half_pts * 1.8,
                    height=frame_half_pts * 2,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.5,
                    transform=trans,
                    clip_on=False,
                )
                ax.add_patch(rect)
            # Draw concentric ellipses
            n_layers = 15
            for i in range(n_layers):
                alpha = (i + 1) / n_layers
                scale = 1 - (i / n_layers) * 0.8
                el = Ellipse(
                    xy=(0, 0),
                    width=size_ * scale * ratio,
                    height=size_ * scale,
                    angle=angle_leg,
                    facecolor="red",
                    alpha=alpha * 0.25,
                    edgecolor="none",
                    transform=trans,
                    clip_on=False,
                )
                ax.add_patch(el)
            # Core
            core_el = Ellipse(
                xy=(0, 0),
                width=size_ * 0.2 * ratio,
                height=size_ * 0.2,
                angle=angle_leg,
                facecolor="red",
                alpha=0.6,
                edgecolor="none",
                transform=trans,
                clip_on=False,
            )
            ax.add_patch(core_el)
