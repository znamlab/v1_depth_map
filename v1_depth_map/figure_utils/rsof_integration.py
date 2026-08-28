"""
Helper functions to plot RSOF integration figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Ellipse, Rectangle
from scipy import stats
from cottage_analysis.plotting import rsof_plots, depth_selectivity_plots
from cottage_analysis.analysis.fit_gaussian_blob import get_gaussian_angle


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
        ylabel="Optic flow speed (°/s)",
        cbar_width=0.01,
        fontsize_dict=fontsize_dict,
    )

    # 1 example neuron, fits of 5 models
    for imodel, (model, model_label) in enumerate(zip(models, model_labels)):
        if imodel == 0:
            ylabel = "Optic flow speed (°/s)"
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
    ax,
    df,
    fontsize_dict,
    ticks=[0.01, 0.1, 1, 10, 100],
    plot_stats=True,
    plot_fit=True,
    plot_unity=False,
    stat_type="pearson",
    suffix="",
    depth_suffix=None,
    **kwargs,
):
    """
    Plots the expected treadmill depth (from preferred RS and OF ratio)
    vs preferred depth in closed loop.

    Parameters
    ----------
    suffix : str
        Suffix for RS and OF columns. The onset-detection method is always explicit:
        '_treadmill_trial_average_plateau' (the default the figures use) or
        '_treadmill_trial_average_model' for trial-averaged fits, '_treadmill' or
        '_treadmill_model' for per-frame fits. Note that bare '_treadmill' IS the plateau
        family -- it kept its name because it is read at ~240 hardcoded sites.
    depth_suffix : str or None
        Suffix for preferred depth column ('_treadmill' or its identical twin
        '_treadmill_plateau', else '_treadmill_model'). If None, defaults to `suffix`.
        Decoupling is required because 1D depth fits are always trial-averaged, so only
        the onset-detection method distinguishes them -- there is no depth equivalent of
        the '_trial_average' tag, and passing an RS/OF suffix here would not resolve.
    """
    if depth_suffix is None:
        depth_suffix = suffix

    ticks = np.asarray(ticks)
    # Data is natively in metres: for the gaussian_2d fit preferred_RS is stored
    # in m/s and preferred_OF in rad/s, so RS/OF is in m; preferred_depth is also
    # in m (converted from cm in spheres.py). Multiply by 100 to display in cm,
    # matching the axis labels and tick values below.
    M_TO_CM = 100
    expected_depth = (
        df[f"preferred_RS_closedloop_crossval_g2d{suffix}"]
        / df[f"preferred_OF_closedloop_crossval_g2d{suffix}"]
    ) * M_TO_CM
    treadmill_depth = df[f"preferred_depth_closedloop_crossval{depth_suffix}"] * M_TO_CM

    x = np.log(expected_depth)
    y = np.log(treadmill_depth)

    sc = ax.scatter(x, y, **kwargs)

    if plot_stats:
        # Remove NaNs if any
        mask = ~np.isnan(x) & ~np.isnan(y)

        if stat_type == "pearson":
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x[mask], y[mask]
            )
            r_str = f"$r^2$ = {r_value**2:.2f}"
            p_val_to_format = p_value
        elif stat_type == "spearman":
            r_val, p_val = stats.spearmanr(x[mask], y[mask])
            r_str = f"Spearman $r$ = {r_val:.2f}"
            p_val_to_format = p_val
            # Still need slope for the fit line if requested
            if plot_fit:
                slope, intercept, _, _, _ = stats.linregress(x[mask], y[mask])

        if p_val_to_format < 0.001:
            exponent = int(np.floor(np.log10(p_val_to_format)))
            coeff = p_val_to_format / 10**exponent
            p_str = f"$p = {coeff:.1f} \\times 10^{{{exponent}}}$"
        else:
            p_str = f"$p = {p_val_to_format:.3f}$"

        stats_text = f"{r_str}\n{p_str}"

        # Place text on the plot
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=fontsize_dict.get("legend", 12),
        )

        # Plot the trendline
        if plot_fit:
            # Use a wide range to ensure it covers the plot area
            x_range = np.array([-10, 10])
            ax.plot(x_range, intercept + slope * x_range, color="grey", linestyle="--")

    if plot_unity:
        x_range = np.array([-10, 10])
        ax.plot(x_range, x_range, color="grey", linestyle=":", zorder=-1)

    ax.set_aspect("equal")
    ax.set_ylabel(
        "Preferred depth during\nfree locomotion (cm)", fontsize=fontsize_dict["label"]
    )
    ax.set_xlabel(
        "Ratio of preferred RS and OF\nwith motorized wheel (cm)",
        fontsize=fontsize_dict["label"],
    )
    # Place ticks at the presented depth values
    ax.set_xticks(np.log(ticks))
    ax.set_xticklabels(ticks)
    ax.set_yticks(np.log(ticks))
    ax.set_yticklabels(ticks)
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])

    return sc, {"x": x, "y": y}


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
    **kwargs,
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
        ylabel="Optic flow speed (°/s)",
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
        angle = get_gaussian_angle(popt)
        if not np.isnan(angle):
            thetas.append(angle)

    if len(thetas) == 0:
        ax.text(0.5, 0.5, "No theta data", ha="center")
        return

    thetas_deg = thetas  # already in degrees

    ax.hist(thetas_deg, bins=30, color="orange", alpha=0.7, edgecolor="k")
    ax.set_xlabel("Gaussian fit angle (degrees)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Number of neurons", fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"])

    return ax


# Perpendicular offsets from the -45 deg radial spoke, as a fraction of the radial axis
# range. The insets sit just outside the radial tick labels, the axis label just outside
# the insets.
ECC_AXIS_INSET_ARC = 0.20
ECC_AXIS_LABEL_ARC = 0.45

# Radial scales the polar RS/OF panels can use. Each entry says how far the axis runs,
# how to tick it, which shapes the legend insets illustrate, and - crucially - how to turn
# a radial coordinate into an ellipse's minor/major axis ratio, so `add_ellipse_schematics`
# draws the shape that actually belongs at that radius whatever the scale.
#
#   "eccentricity": r = sqrt(1 - b^2/a^2), the standard geometric eccentricity. Saturates
#       hard - a 2:1 ellipse is already at 0.87 and a 4:1 at 0.97 - so real populations
#       pile up against the outer ring.
#   "elongation": r = log2(a/b). Zero is circular and every unit is a doubling, which is
#       the scale elongation actually varies on, so the same population spreads out. Ridge
#       fits are unbounded, hence the clip at the outermost tick: everything at 8:1 or
#       beyond lands on the outer ring.
#
# `size_mode` says which axis of the legend insets is held constant as the shape changes:
#   "fixed_major": the major axis fills the frame and the minor shrinks. Reads as
#       "same length, squeezed".
#   "fixed_minor": the minor axis is the same in every inset and the major grows out of
#       it. Reads as "a circle stretched along one axis", which is what the elongation
#       axis actually measures - so the staircase starts from a circle at 1:1.
RADIAL_SCALES = {
    "eccentricity": dict(
        rmax=1.0,
        ticks=(0, 0.2, 0.4, 0.6, 0.8, 1.0),
        ticklabels=("0", "0.2", "0.4", "0.6", "0.8", "1"),
        label="Eccentricity",
        legend_values=(0.2, 0.4, 0.6, 0.8),
        legend_radii=(0.26, 0.44, 0.62, 0.80),
        # r is the eccentricity itself -> b/a = sqrt(1 - r^2)
        ratio=lambda v: float(np.sqrt(max(1.0 - float(v) ** 2, 0.0))),
        size_mode="fixed_major",
    ),
    "elongation": dict(
        rmax=3.0,
        ticks=(0, 1, 2, 3),
        ticklabels=("1:1", "2:1", "4:1", "\u22658:1"),
        label="Elongation",
        legend_values=(0, 1, 2, 3),
        legend_radii=(0.0, 1.05, 2.02, 2.88),
        # r = log2(a/b) -> b/a = 2**-r
        ratio=lambda v: float(2.0 ** -float(v)),
        size_mode="fixed_minor",
        legend_minor_boost=3.0,
        # With the minor axis held constant and boosted, the most elongated insets are
        # longer than their frame; cut them off at the edge rather than let them spill
        # over the panel. The outermost tick is "at least 8:1", so an inset running off
        # the edge of its box reads correctly.
        legend_clip=True,
        # Insets sit exactly perpendicular to their tick, so `legend_radii` are distances
        # along the spoke rather than plot radii. Needed for the 1:1 inset, which is at
        # the origin, where the small-angle offset of `_offset_theta` blows up.
        legend_exact=True,
    ),
}


def _get_radial_scale(radial_scale):
    """Look up a radial-scale spec by name, with a helpful error."""
    try:
        return RADIAL_SCALES[radial_scale]
    except KeyError:
        raise ValueError(
            f"Unknown radial_scale {radial_scale!r}; expected one of "
            f"{tuple(RADIAL_SCALES)}."
        ) from None


def _offset_theta(r, arc, base_deg=-45):
    """Polar angle of a point `arc` away from the `base_deg` spoke, at radius `r`.

    Used to lay text and ellipse insets alongside the eccentricity axis at a roughly
    constant perpendicular distance from it, rather than at a constant angle (which
    would bunch them up near the origin).

    Args:
        r (float): Radius of the point, in radial-axis units.
        arc (float): Perpendicular distance from the spoke, in radial-axis units.
        base_deg (float, optional): Angle of the spoke in degrees. Default is -45.

    Returns:
        float: Angle in degrees.
    """
    return base_deg - np.degrees(arc / r)


def _spoke_offset_point(along, arc, base_deg=-45):
    """Exact polar coordinates of a point beside the `base_deg` spoke.

    Unlike `_offset_theta`, which approximates the offset as an arc, this treats
    (`along`, `arc`) as Cartesian coordinates in the spoke's own frame. It is therefore
    valid all the way down to `along = 0` (the origin), where the arc approximation
    diverges.

    Args:
        along (float): Distance from the origin along the spoke, in radial-axis units.
        arc (float): Perpendicular distance from the spoke, in radial-axis units.
        base_deg (float, optional): Angle of the spoke in degrees. Default is -45.

    Returns:
        tuple[float, float]: Angle in degrees and radius in radial-axis units.
    """
    return (
        base_deg - np.degrees(np.arctan2(arc, along)),
        float(np.hypot(along, arc)),
    )


def _legend_axes_pts(spec, size_pts, angle_deg):
    """Major-axis length in points for each legend inset of a radial scale.

    With `size_mode="fixed_major"` every inset has the same major axis and the minor
    axis shrinks with the ratio. With `"fixed_minor"` it is the minor axis that is
    shared, and the major grows as `minor / ratio`. The reference minor is the one that
    would make the longest inset just fill the frame, scaled by `legend_minor_boost`; a
    boost above 1 therefore trades insets that run past their frame for a minor axis
    thick enough to see.

    Args:
        spec (dict): Entry of `RADIAL_SCALES`.
        size_pts (float): Largest axis length that fits the frame, in points.
        angle_deg (float): Rotation the insets are drawn at, in degrees. Only the
            bounding box of the rotated ellipse depends on it.

    Returns:
        list[float]: Major axis in points, one per entry of `spec["legend_values"]`.
    """
    ratios = [spec["ratio"](v) for v in spec["legend_values"]]
    if spec.get("size_mode", "fixed_major") == "fixed_major":
        return [size_pts] * len(ratios)
    # Half-extent of an ellipse (major m_a, minor m_a * ratio) rotated by `angle_deg`,
    # as a fraction of its major axis. The frame is axis-aligned and square, so the
    # binding constraint is the larger of the two half-extents.
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    r_min = min(ratios)
    half_x = np.hypot(r_min * c, s) / 2
    half_y = np.hypot(r_min * s, c) / 2
    major_max = size_pts / (2 * max(half_x, half_y))
    # A minor axis that just fits the longest inset in the frame is hairline-thin (the
    # legend spans a factor of 8), so it is boosted to stay legible in print and the
    # longest insets are allowed to run past their frame.
    minor_pts = major_max * r_min * spec.get("legend_minor_boost", 1.0)
    return [minor_pts / ratio for ratio in ratios]


# Every schematic frame is drawn under every schematic ellipse, and both under the
# scatter (which uses zorder 3), whatever order the insets are added in.
FRAME_ZORDER = 1.0
ELLIPSE_ZORDER = 1.5
# Range the per-inset bands of `rasterized="each"` are spread over. It has to avoid the
# axes' own artists entirely: any non-rasterized artist landing inside a band splits that
# inset across two images, which is exactly what the mode exists to prevent. A polar axes
# puts ThetaAxis and RadialAxis at 1.5 and the spines at 2.5, so the bands sit between
# them. The insets are drawn outside the wedge, so being above the grid rather than below
# it (as in single-layer mode) makes no visible difference.
SCHEMATIC_ZORDER_BOTTOM = 1.55
SCHEMATIC_ZORDER_TOP = 2.45


def _draw_gradient_ellipse(
    ax,
    trans,
    major_pts,
    ratio,
    angle,
    color="red",
    frame=True,
    frame_half_pts=10.0,
    clip_to_frame=False,
    rasterized=False,
    frame_zorder=FRAME_ZORDER,
    ellipse_zorder=ELLIPSE_ZORDER,
    flush_zorder=None,
):
    """Draw one soft-edged ellipse schematic on a white square frame.

    The "gradient" is `n_layers` concentric alpha-ramped ellipses plus a sharp core.
    Everything is drawn in points around the origin of `trans`, so the shape is
    independent of the axes size and of the polar coordinates it sits in.

    Args:
        ax (matplotlib.axes.Axes): Axes to add the patches to.
        trans (matplotlib.transforms.Transform): Transform mapping points-from-origin to
            display coordinates (i.e. `Affine2D().scale(dpi/72) + ScaledTranslation(...)`).
        major_pts (float): Length of the major axis, in points.
        ratio (float): Minor/major axis ratio, i.e. `sqrt(1 - eccentricity**2)` with
            the standard geometric definition of eccentricity.
        angle (float): Rotation of the ellipse in degrees. The major axis is `height`,
            so it points along +y at `angle=0`.
        color (str, optional): Fill colour. Default is "red".
        frame (bool, optional): Whether to draw the white square frame behind the
            ellipse. Default is True.
        frame_half_pts (float, optional): Half-side of the square frame, in points.
            Default is 10.
        clip_to_frame (bool, optional): Whether to clip the ellipse to the frame, so an
            ellipse longer than its frame is cut off at the edge rather than running
            past it. Ignored when `frame` is False. Default is False.
        rasterized (bool, optional): Whether to rasterize the patches. Default is False.
        frame_zorder (float, optional): zorder of the frame. Defaults to `FRAME_ZORDER`.
        ellipse_zorder (float, optional): zorder of the gradient layers. Defaults to
            `ELLIPSE_ZORDER`.
        flush_zorder (float, optional): If given, an invisible, non-rasterized artist is
            added at this zorder, just above the inset. Matplotlib merges *consecutive*
            rasterized artists into one image, so this is what ends the group and makes
            the inset its own image in the output. Default is None (no separator).
    """
    # Explicit zorders rather than draw order: an ellipse longer than its own frame would
    # otherwise be painted over by the white frame of the next inset along.
    box = Rectangle(
        xy=(-frame_half_pts, -frame_half_pts),
        width=frame_half_pts * 2,
        height=frame_half_pts * 2,
        facecolor="white",
        edgecolor="black",
        linewidth=0.5,
        transform=trans,
        clip_on=False,
        zorder=frame_zorder,
        rasterized=rasterized,
    )
    if frame:
        ax.add_patch(box)
    # The frame is drawn with a stroke centred on its edge, so clipping to the patch cuts
    # the ellipse at the middle of the border line and leaves it looking contained.
    clip = frame and clip_to_frame
    n_layers = 15

    def add_layer(scale_el, alpha):
        patch = Ellipse(
            xy=(0, 0),
            width=major_pts * scale_el * ratio,
            height=major_pts * scale_el,
            angle=angle,
            facecolor=color,
            alpha=alpha,
            edgecolor="none",
            transform=trans,
            clip_on=clip,
            zorder=ellipse_zorder,
            rasterized=rasterized,
        )
        # The path and its transform, not the `box` patch: `set_clip_path` special-cases
        # a Rectangle into a clip *box* and leaves the clip *path* unset, which then lets
        # `add_patch` fill it in with the polar wedge - and the insets sit outside it, so
        # they would vanish entirely.
        if clip:
            patch.set_clip_path(box.get_path(), box.get_transform())
        ax.add_patch(patch)

    for i in range(n_layers):
        add_layer(1 - (i / n_layers) * 0.8, (i + 1) / n_layers * 0.25)
    add_layer(0.2, 0.6)  # core, for sharpness
    if flush_zorder is not None:
        ax.add_line(Line2D([], [], visible=False, zorder=flush_zorder))


def add_ellipse_schematics(
    ax,
    plot_angle=True,
    plot_ecc=True,
    frame=True,
    scale=1.0,
    rasterized=False,
    color="red",
    perimeter_ratio=0.4,
    radial_scale="eccentricity",
):
    """
    Add oriented, gradient-filled ellipses to a polar plot to visualize tuning markers and radial scale.

    Each legend inset's shape is derived from the radius it sits at, via the active
    `radial_scale`'s `ratio` function, so an inset next to a radial tick really has the
    shape that tick denotes - whether the axis is eccentricity or log2 elongation. Which
    axis of the inset is held constant as the shape changes is the scale's `size_mode`:
    "fixed_major" squeezes the minor axis in, "fixed_minor" (used by "elongation")
    stretches the major axis out of a common circle. The frames are the same square in
    both cases, so they read as a ruler for the axis that is held constant - under
    "fixed_minor" the most elongated insets deliberately run past theirs.

    Args:
        ax (matplotlib.axes.PolarAxes): The polar axes to which the ellipses will be added.
        plot_angle (bool, optional): Whether to plot the orientation ellipses around the
            perimeter. Default is True.
        plot_ecc (bool, optional): Whether to plot the shape legend ellipses along the
            radial axis. Default is True.
        frame (bool, optional): Whether to draw a white rectangle with a thin black border
            around each ellipse. Default is True.
        scale (float, optional): Scaling factor for the size of the ellipses and frames.
            Default is 1.0.
        rasterized (bool or str, optional): Whether to rasterize the ellipses and frames.
            False (default) leaves them vector; True puts all of them in one raster layer;
            "each" gives every inset its own raster layer, so the output carries one image
            per box and the boxes can be moved independently in a vector editor. "each"
            also gives every inset its own zorder band rather than drawing all the frames
            below all the ellipses, so an inset that overflowed its frame could be painted
            over by the next one along - at the default sizes none do.
        color (str, optional): Fill colour of the ellipses. Default is "red".
        perimeter_ratio (float, optional): Minor/major axis ratio of the orientation
            ellipses around the perimeter. Default is 0.4, i.e. 2.5:1 - still clearly
            oriented, but thick enough to read as an ellipse rather than a line at small
            sizes.
        radial_scale (str, optional): Key into `RADIAL_SCALES`, sets the radial extent and
            the shape of the legend insets. Default is "eccentricity".
    """
    spec = _get_radial_scale(radial_scale)
    rmax = spec["rmax"]
    fig = ax.get_figure()
    # Points to display units; shared by both groups so `scale` behaves consistently.
    # `dpi_scale_trans` (inches -> display) rather than a baked-in `fig.dpi / 72`: the
    # dpi is not the same at draw time as it was here - the vector backends draw at 72
    # and a rasterized artist at the savefig dpi - so a frozen factor makes the insets
    # come out a different physical size in every output.
    point_to_pixel = transforms.Affine2D().scale(1 / 72) + fig.dpi_scale_trans
    frame_half_pts = 10 * scale  # half-side of the square frame
    # Major axis, sized to fill the frame with a small margin
    size_pts = frame_half_pts * 1.6

    # With rasterized="each", each inset needs its own zorder band and a separator above
    # it: matplotlib merges consecutive rasterized artists into one image, so without a
    # non-rasterized artist between them all the insets end up in a single image.
    per_inset = rasterized == "each"
    n_insets = (5 if plot_angle else 0) + (
        len(spec["legend_values"]) if plot_ecc else 0
    )
    band = (SCHEMATIC_ZORDER_TOP - SCHEMATIC_ZORDER_BOTTOM) / max(n_insets, 1)
    inset_count = 0

    def next_zorders():
        """(frame, ellipse, separator) zorders for the next inset."""
        nonlocal inset_count
        if not per_inset:
            return FRAME_ZORDER, ELLIPSE_ZORDER, None
        base = SCHEMATIC_ZORDER_BOTTOM + inset_count * band
        inset_count += 1
        return base, base + 0.4 * band, base + 0.8 * band

    if plot_angle:
        # PLOT ANGLE ELLIPSES
        # One inset per angular tick, showing the tuning ellipse that produces that
        # orientation. `Ellipse` puts the major axis (height) along +y at angle=0, so
        # `angle = theta - 90` puts it at `theta` degrees from horizontal: vertical at
        # 90 deg, horizontal at 0 deg.
        ratio = perimeter_ratio
        # Just outside the radial limit, beyond the angular tick labels
        r_pos = 1.25 * rmax
        for theta_deg in [-45, 0, 45, 90, 135]:
            trans = point_to_pixel + transforms.ScaledTranslation(
                np.radians(theta_deg), r_pos, ax.transData
            )
            z_frame, z_ellipse, z_flush = next_zorders()
            _draw_gradient_ellipse(
                ax,
                trans,
                major_pts=size_pts,
                ratio=ratio,
                angle=theta_deg - 90,
                color=color,
                frame=frame,
                frame_half_pts=frame_half_pts,
                rasterized=bool(rasterized),
                frame_zorder=z_frame,
                ellipse_zorder=z_ellipse,
                flush_zorder=z_flush,
            )

    if plot_ecc:
        # PLOT SHAPE LEGEND ELLIPSES
        # Legend staircase running alongside the radial (-45 deg) axis: same orientation
        # throughout, only the elongation changes. The offsets are fractions of the radial
        # range so the layout is identical whatever `rmax` the scale uses.
        # Drawn at a tuning angle of 45 deg, as the perimeter insets are: same
        # `theta - 90` convention, so the major axis points 45 deg up from horizontal.
        angle_leg = 45 - 90
        inset_arc = ECC_AXIS_INSET_ARC * rmax
        majors = _legend_axes_pts(spec, size_pts, angle_leg)
        for value, r_pos, major_pts in zip(
            spec["legend_values"], spec["legend_radii"], majors
        ):
            if spec.get("legend_exact", False):
                theta_deg, r_draw = _spoke_offset_point(r_pos, inset_arc)
            else:
                theta_deg, r_draw = _offset_theta(r_pos, inset_arc), r_pos
            trans = point_to_pixel + transforms.ScaledTranslation(
                np.radians(theta_deg),
                r_draw,
                ax.transData,
            )
            z_frame, z_ellipse, z_flush = next_zorders()
            _draw_gradient_ellipse(
                ax,
                trans,
                major_pts=major_pts,
                ratio=spec["ratio"](value),
                angle=angle_leg,
                color=color,
                frame=frame,
                frame_half_pts=frame_half_pts,
                clip_to_frame=spec.get("legend_clip", False),
                rasterized=bool(rasterized),
                frame_zorder=z_frame,
                ellipse_zorder=z_ellipse,
                flush_zorder=z_flush,
            )


def plot_angle_eccentricity_polar(
    ax,
    theta_deg,
    eccentricity,
    fontsize_dict,
    scale=1.0,
    schematics=True,
    ecc_label=None,
    radial_scale="eccentricity",
    axis_ratio=None,
    rasterize_schematics=False,
    **scatter_kwargs,
):
    """Polar scatter of g2d tuning-ellipse orientation against its shape.

    Sets up the -45 to 135 degree wedge used throughout the RS/OF analyses (0 deg = optic
    flow axis, 90 deg = running speed axis), scatters the neurons and adds the ellipse
    schematics.

    The radial axis is either the geometric eccentricity or the log2 elongation, see
    `RADIAL_SCALES`. Eccentricity saturates - most well-fit neurons sit above 0.9 - so
    "elongation" is usually the readable choice for a population.

    Args:
        ax (matplotlib.axes.PolarAxes): Polar axes to draw on.
        theta_deg (array-like): Ellipse major-axis angle in degrees, wrapped to
            [-45, 135] (as returned by `fit_gaussian_blob.get_gaussian_angle`).
        eccentricity (array-like): Ellipse eccentricity,
            `sqrt(1 - sigma_minor^2/sigma_major^2)`. Used directly when
            `radial_scale="eccentricity"`; for "elongation" it is only a fallback for
            `axis_ratio` (pass `axis_ratio` instead, it is numerically better near 1).
        fontsize_dict (dict): Font sizes, keys "label" and "tick".
        scale (float, optional): Scaling factor passed to `add_ellipse_schematics`.
            Default is 1.0.
        schematics (bool, optional): Whether to add the ellipse schematics. Default is
            True.
        ecc_label (str, optional): Label of the radial axis. Defaults to the active
            scale's own label.
        radial_scale (str, optional): Key into `RADIAL_SCALES`. Default is
            "eccentricity".
        axis_ratio (array-like, optional): Ellipse `sigma_major / sigma_minor` (>= 1).
            Only used by `radial_scale="elongation"`, where it is preferred over
            deriving the ratio from `eccentricity`.
        rasterize_schematics (bool or str, optional): Whether to rasterize the ellipse
            schematics only. Each is ~16 alpha-blended patches, so a panel carries a few
            hundred overlapping translucent shapes that bloat a PDF/SVG and are slow to
            open in a vector editor; rasterizing them leaves the scatter, the axes and
            all the text as vector. True merges them into one image, "each" gives every
            inset its own image so they stay individually selectable and movable in a
            vector editor. Passed to `add_ellipse_schematics` as `rasterized`. Default is
            False.
        **scatter_kwargs: Passed to `ax.scatter`. Note `rasterized` here applies to the
            scatter, not the schematics.

    Returns:
        matplotlib.collections.PathCollection: The scatter artist.
    """
    spec = _get_radial_scale(radial_scale)
    rmax = spec["rmax"]

    if radial_scale == "eccentricity":
        r = np.asarray(eccentricity, dtype=float)
    else:  # "elongation": radius is log2 of the axis ratio
        if axis_ratio is None:
            if eccentricity is None:
                raise ValueError("Provide either `axis_ratio` or `eccentricity`.")
            # a/b = 1 / sqrt(1 - e^2); loses precision as e -> 1, hence the preference
            # for an explicit `axis_ratio`
            e = np.asarray(eccentricity, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                axis_ratio = 1.0 / np.sqrt(np.clip(1.0 - e**2, 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.log2(np.asarray(axis_ratio, dtype=float))
    # Ridge fits are unbounded; park them on the outer ring rather than dropping them
    r = np.clip(r, 0, rmax)

    # zorder above the spines: clipped/ridge fits land exactly on the outer arc and would
    # otherwise be hidden behind it
    kwargs = dict(s=10, alpha=0.4, linewidths=0, clip_on=False, zorder=3)
    kwargs.update(scatter_kwargs)
    sc = ax.scatter(np.radians(np.asarray(theta_deg, dtype=float)), r, **kwargs)

    ax.set_theta_zero_location("E")  # 0 is East (Right) -> OF
    ax.set_thetalim(np.radians(-45), np.radians(135))
    ax.set_xticks(np.radians([-45, 0, 45, 90, 135]))
    ax.set_xticklabels(
        ["-45°", "0°", "45°", "90°", "135°"], fontsize=fontsize_dict["tick"]
    )
    ax.tick_params(axis="both", labelsize=fontsize_dict["tick"], pad=0)
    ax.set_rlim(0, rmax)
    ax.set_rticks(list(spec["ticks"]))
    ax.set_yticklabels(list(spec["ticklabels"]), fontsize=fontsize_dict["tick"])
    # Radial labels along the radial axis rather than the 0 deg spoke
    ax.set_rlabel_position(-45)
    r_label = 0.57 * rmax
    ax.text(
        np.radians(_offset_theta(r_label, ECC_AXIS_LABEL_ARC * rmax)),
        r_label,
        spec["label"] if ecc_label is None else ecc_label,
        rotation=-45,
        ha="center",
        va="center",
        fontsize=fontsize_dict["label"],
    )
    if schematics:
        add_ellipse_schematics(
            ax,
            scale=scale,
            radial_scale=radial_scale,
            rasterized=rasterize_schematics,
        )
    return sc


# Colours of the two principal axes drawn by `plot_g2d_fit_schematic`. Exported so a
# figure can colour-code the "sigma_major / sigma_minor" text of its legend to match.
SEMIMAJOR_COLOR = "#0072B2"
SEMIMINOR_COLOR = "#009E73"


def plot_g2d_fit_schematic(
    ax,
    neurons_df,
    roi,
    sfx,
    min_sigma,
    fontsize_dict,
    mass_fraction=0.5,
    semimajor_color=SEMIMAJOR_COLOR,
    semiminor_color=SEMIMINOR_COLOR,
    draw_theta=True,
    ticks=False,
    **fit_kwargs,
):
    """Draw one neuron's 2D-Gaussian RS/OF fit, outlined with what the fit measures.

    On top of the fitted surface (`rsof_plots.plot_RS_OF_fit`) this adds the iso-response
    ellipse enclosing `mass_fraction` of the Gaussian's mass, its two principal semi-axes
    colour-coded major/minor, and the angle theta between the major axis and the optic
    flow axis. Intended as a schematic of the fit parameters rather than a data panel,
    hence `ticks=False` by default.

    `plot_RS_OF_fit` draws in log_base units of the *displayed* quantities (x =
    log_base(RS in cm/s), y = log_base(OF in deg/s)) while the fit itself lives in natural
    log of RS in m/s and OF in deg/s. Both axes rescale by the same 1/ln(base), so the
    ellipse keeps its shape and theta is unchanged - only the centre and the sigmas need
    converting. imshow sets aspect="equal", so the drawn angle is the true one.

    Args:
        ax (matplotlib.axes.Axes): Axes to draw on.
        neurons_df (pd.DataFrame): Population dataframe, or any subset containing the
            neuron. It must be narrowed to a single session first: the ROI is looked up by
            `neurons_df.roi == roi`, which is not unique across sessions.
        roi (int): ROI number of the neuron to draw.
        sfx (str): Column suffix of the fit family, e.g.
            "_treadmill_trial_average_plateau".
        min_sigma (float): `min_sigma` the fit was run with, as returned by
            `treadmill.add_trial_average_rsof_columns`.
        fontsize_dict (dict): Font sizes, keys "label" and "tick".
        mass_fraction (float, optional): Fraction of the Gaussian's mass the drawn
            ellipse encloses. Raise it to widen the ellipse. Defaults to 0.5.
        semimajor_color (str, optional): Colour of the semi-major axis.
        semiminor_color (str, optional): Colour of the semi-minor axis.
        draw_theta (bool, optional): Whether to add the horizontal reference, the arc and
            the theta label. Defaults to True.
        ticks (bool, optional): Whether to keep the axis ticks. Defaults to False.
        **fit_kwargs: Passed to `rsof_plots.plot_RS_OF_fit`, overriding the schematic
            defaults below. Pass `log_range`, `rs_bins`, `of_bins` and `tick_dict` here to
            match the data matrices of the same figure.

    Returns:
        dict: The drawn geometry in display units - "centre" (x, y), "sigma_major",
            "sigma_minor" (the scaled sigmas, not multiplied by the mass-fraction radius),
            "theta_deg" (major axis, in [0, 180)) and "k_iso" (the Mahalanobis radius the
            ellipse is drawn at).
    """
    fit_kwargs = dict(
        model_label="Gaussian fit",
        label_r2=False,
        cbar_width=None,
        xlabel="Running speed",
        ylabel="Optic flow",
        **fit_kwargs,
    )
    rsof_plots.plot_RS_OF_fit(
        neurons_df=neurons_df,
        roi=roi,
        model="g2d",
        sfx=sfx,
        min_sigma=min_sigma,
        fontsize_dict=fontsize_dict,
        ax=ax,
        **fit_kwargs,
    )
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    # Adding artists lets autoscaling push the limits past the imshow extent, which would
    # shrink the matrix in the panel. Hold the limits the image set and restore them after.
    fit_xlim, fit_ylim = ax.get_xlim(), ax.get_ylim()

    popt = neurons_df.loc[neurons_df.roi == roi, f"rsof_popt_closedloop_g2d{sfx}"].iloc[
        0
    ]
    _, fit_x0, fit_y0, log_sigma_x2, log_sigma_y2, fit_theta = popt[:6]
    sigma_x = np.sqrt(np.exp(log_sigma_x2) + min_sigma)
    sigma_y = np.sqrt(np.exp(log_sigma_y2) + min_sigma)

    ln_base = np.log(fit_kwargs.get("log_range", {}).get("log_base", 10))
    centre_x = (fit_x0 + np.log(100)) / ln_base  # m/s -> cm/s, then into log_base units
    centre_y = fit_y0 / ln_base
    sx, sy = sigma_x / ln_base, sigma_y / ln_base

    # Mahalanobis radius enclosing a fraction f of a 2D Gaussian's mass: sqrt(-2*ln(1-f))
    k_iso = np.sqrt(-2 * np.log(1 - mass_fraction))

    ax.add_patch(
        Ellipse(
            (centre_x, centre_y),
            width=2 * k_iso * sx,
            height=2 * k_iso * sy,
            angle=np.degrees(fit_theta),
            facecolor="none",
            edgecolor="k",
            linewidth=1,
            clip_on=True,
        )
    )

    # sigma_x lies along theta, so whichever sigma is larger defines the major axis. Both
    # segments run from the centre out to the ellipse, so they read as the semi-axes.
    if sx >= sy:
        axes_spec = ((fit_theta, sx), (fit_theta + np.pi / 2, sy))
    else:
        axes_spec = ((fit_theta + np.pi / 2, sy), (fit_theta, sx))
    for (direction, half_length), color in zip(
        axes_spec, (semimajor_color, semiminor_color)
    ):
        ax.plot(
            [centre_x, centre_x + k_iso * half_length * np.cos(direction)],
            [centre_y, centre_y + k_iso * half_length * np.sin(direction)],
            color=color,
            linewidth=1.2,
            clip_on=True,
        )

    major_dir, major_len = axes_spec[0]
    theta_draw_deg = np.degrees(major_dir) % 180
    if draw_theta:
        # Horizontal reference through the centre, plus the arc between it and the major
        # axis, so theta can be read straight off the panel. The panel is narrow (the
        # image is aspect-locked) and the arc has to sit clear of the ellipse outline, so
        # both the arc and the label are sized off the ellipse's own radius along the
        # bisector rather than off the semi-major axis.
        bisector = np.radians(theta_draw_deg / 2)
        d_bisector = bisector - fit_theta
        r_edge = 1.0 / np.hypot(
            np.cos(d_bisector) / (k_iso * sx), np.sin(d_bisector) / (k_iso * sy)
        )
        ax.plot(
            [centre_x, centre_x + 0.65 * k_iso * major_len],
            [centre_y, centre_y],
            color="k",
            ls="--",
            linewidth=0.8,
            clip_on=True,
        )
        arc_r = 0.42 * r_edge
        ax.add_patch(
            Arc(
                (centre_x, centre_y),
                width=2 * arc_r,
                height=2 * arc_r,
                theta1=0,
                theta2=theta_draw_deg,
                edgecolor="k",
                linewidth=0.8,
            )
        )
        # label on the bisector, between the arc and the ellipse outline
        ax.text(
            centre_x + 0.72 * r_edge * np.cos(bisector),
            centre_y + 0.72 * r_edge * np.sin(bisector),
            r"$\theta$",
            fontsize=fontsize_dict["label"],
            ha="center",
            va="center",
        )

    ax.set_xlim(fit_xlim)
    ax.set_ylim(fit_ylim)
    return dict(
        centre=(centre_x, centre_y),
        sigma_major=max(sx, sy),
        sigma_minor=min(sx, sy),
        theta_deg=theta_draw_deg,
        k_iso=k_iso,
    )
