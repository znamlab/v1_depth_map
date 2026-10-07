"""Linear gradients of RF azimuth, elevation and preferred depth across each V1 FOV.

For every single-depth V1 session, fits each map as a plane over the ROI centres
of the selected neurons (depth-tuned, significant RF, as in Figure 4 I-K) with
`roi_location.spatial_gradient`. The R2 says how much of the map a smooth
gradient explains; the shuffle p-value tests it against cells randomly
reassigned to positions. Neighbouring cells are spatially correlated, so those
p-values are optimistic.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from cottage_analysis.analysis import roi_location
from v1_depth_map.figure_utils.rf_distance import PIXEL_SIZE, load_session_neurons

# column, transform, label
MAPS = [
    ("rf_azi", None, "Azimuth"),
    ("rf_ele", None, "Elevation"),
    ("preferred_depth_closedloop", np.log2, "Depth"),
]
MAP_COLORS = ["#d95f02", "#1b9e77", "#7570b3"]


def fov_gradients_per_session(
    flexilims_session, session_name, rf_file=None, min_neurons=10, n_perm=10000
):
    """One row per map with the gradient fit of a session.

    Sessions with fewer than `min_neurons` selected neurons get no rows.
    `magnitude` is in map units per 100 um (log2 units for depth), `direction`
    in degrees in image coordinates (x right, y down).
    """
    neurons_df, select_neurons = load_session_neurons(
        flexilims_session, session_name, rf_file=rf_file
    )
    cells = neurons_df[select_neurons]
    if len(cells) < min_neurons:
        return pd.DataFrame()
    rows = []
    for col, transform, label in MAPS:
        values = cells[col].to_numpy(float)
        g = roi_location.spatial_gradient(
            cells["center_x"],
            cells["center_y"],
            transform(values) if transform else values,
            n_perm=n_perm,
        )
        g["magnitude"] = g["magnitude"] / PIXEL_SIZE * 100
        rows.append(dict(session=session_name, map=label, **g))
    return pd.DataFrame(rows)


def load_fov_gradients(
    flexilims_session, cache_path, recompute=False, rf_file=None, min_neurons=10
):
    """Gradient fits of all V1 sessions, one row per session and map.

    With `recompute`, refit every session (RF columns from `rf_file` if given)
    and overwrite the pickle at `cache_path`; otherwise read that pickle.
    """
    cache_path = Path(cache_path)
    if not recompute:
        return pd.read_pickle(cache_path)
    from cottage_analysis.summary_analysis import get_session_list

    session_list = get_session_list.get_sessions(
        flexilims_session=flexilims_session,
        exclude_openloop=False,
        exclude_pure_closedloop=False,
        v1_only=True,
    )
    gradients_df = pd.concat(
        [
            fov_gradients_per_session(
                flexilims_session,
                session_name,
                rf_file=rf_file,
                min_neurons=min_neurons,
            )
            for session_name in session_list
        ],
        ignore_index=True,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gradients_df.to_pickle(cache_path)
    return gradients_df


def plot_fov_gradient_panels(
    fig, hist_rect, prop_rect, gradients_df, fontsize_dict, alpha=0.05
):
    """R2 histograms of the three maps, and the proportion of significant sessions.

    A session is significant when its shuffle p-value is below `alpha`; the
    dashed line on the proportion panel is that chance level.
    """
    ax_hist = fig.add_axes(hist_rect)
    bins = np.arange(0, 1, 0.1)
    for (_, _, label), color in zip(MAPS, MAP_COLORS):
        r2 = gradients_df.loc[gradients_df["map"] == label, "r2"]
        ax_hist.hist(
            r2, bins=bins, color=color, alpha=0.5, label=label, edgecolor="none"
        )
        ax_hist.scatter(
            r2.median(),
            1,
            color=color,
            marker="v",
            s=100,
            linewidth=1,
            transform=ax_hist.get_xaxis_transform(),
        )
    ax_hist.set_xlabel(r"Gradient fit $R^2$", fontsize=fontsize_dict["label"])
    ax_hist.set_ylabel("Number of sessions", fontsize=fontsize_dict["label"])
    ax_hist.tick_params(labelsize=fontsize_dict["tick"])
    ax_hist.legend(fontsize=fontsize_dict["legend"], frameon=False)
    sns.despine(ax=ax_hist)

    ax_prop = fig.add_axes(prop_rect)
    labels = [label for _, _, label in MAPS]
    is_sig = gradients_df.assign(sig=gradients_df["pval_perm"] < alpha).groupby("map")[
        "sig"
    ]
    prop = is_sig.mean().reindex(labels)
    ax_prop.bar(range(len(labels)), prop, color=MAP_COLORS, width=0.6)
    ax_prop.axhline(alpha, color="gray", linestyle="--", linewidth=0.8)
    ax_prop.set_xticks(range(len(labels)))
    ax_prop.set_xticklabels(labels, fontsize=fontsize_dict["tick"], rotation=45)
    ax_prop.set_ylim(0, 1)
    ax_prop.set_yticks([0, 0.5, 1])
    ax_prop.tick_params(axis="y", labelsize=fontsize_dict["tick"])
    ax_prop.set_ylabel(
        "Proportion of sessions\nwith significant gradient",
        fontsize=fontsize_dict["label"],
    )
    sns.despine(ax=ax_prop)
    return ax_hist, ax_prop


def plot_fov_gradient_arrows(
    fig,
    rects,
    gradients_df,
    fontsize_dict,
    alpha=0.05,
    nonsig_color="lightgrey",
    nonsig_alpha=0.4,
):
    """One axes per map with each session's gradient drawn as an arrow from the origin.

    Arrow length is the gradient magnitude (map units per 100 um), in the map
    colour when the shuffle p-value is below `alpha`, light grey otherwise.
    Both axes are inverted to match `plot_example_fov`, which shows the image
    with y down and x flipped, so arrows point as on the FOV maps.
    """
    units = {"Azimuth": "deg", "Elevation": "deg", "Depth": "log$_2$ units"}
    axes = []
    for rect, (_, _, label), color in zip(rects, MAPS, MAP_COLORS):
        ax = fig.add_axes(rect)
        df = gradients_df[gradients_df["map"] == label]
        # slope_x/y are per pixel; scale to per 100 um like `magnitude`
        dx = df["slope_x"].to_numpy() / PIXEL_SIZE * 100
        dy = df["slope_y"].to_numpy() / PIXEL_SIZE * 100
        sig = df["pval_perm"].to_numpy() < alpha
        # non-significant sessions first, so significant arrows sit on top
        for mask, c, a in [(~sig, nonsig_color, nonsig_alpha), (sig, color, 1)]:
            ax.quiver(
                np.zeros(mask.sum()),
                np.zeros(mask.sum()),
                dx[mask],
                dy[mask],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=c,
                alpha=a,
                width=0.012,
                headwidth=3,
                headlength=4,
                headaxislength=3.5,
            )
        lim = np.max(np.hypot(dx, dy)) * 1.05
        ax.set_xlim(lim, -lim)
        ax.set_ylim(lim, -lim)
        ax.set_aspect("equal")
        ax.axhline(0, color="k", linewidth=0.4, zorder=0)
        ax.axvline(0, color="k", linewidth=0.4, zorder=0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"{label}\n{sig.sum()}/{len(sig)} sig.",
            fontsize=fontsize_dict["label"],
            pad=2,
        )
        ax.text(
            0.5,
            -0.02,
            f"Radius: {lim:.2g} {units[label]} / 100 µm",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize_dict["tick"],
        )
        axes.append(ax)
    return axes


def overview_gradients(neurons_df_sig, pixel_size, min_neurons=10, n_perm=10000):
    """Gradient fits of each session in the aligned widefield overview frame.

    Same plane fit as `fov_gradients_per_session`, but on the neuron positions
    aligned across mice for Supplementary Figure 9 (overview_x_aligned,
    overview_y_aligned). The alignment only shifts each mouse, and a plane fit
    is unchanged by a rigid change of coordinates, so R2 and p-values match the
    FOV-pixel fits; the slopes are expressed in cortical orientation.

    Args:
        neurons_df_sig (pd.DataFrame): selected neurons of all sessions, e.g.
            depth-tuned cells with a significant RF.
        pixel_size (float): um per overview pixel.

    Returns:
        pd.DataFrame: one row per session and map, with the session centre
            (x0, y0, overview pixels), slopes (map units per overview pixel),
            magnitude (map units per 100 um) and the `spatial_gradient` outputs.
    """
    rows = []
    for session, cells in neurons_df_sig.groupby("session"):
        cells = cells.dropna(subset=["overview_x_aligned", "overview_y_aligned"])
        if len(cells) < min_neurons:
            continue
        for col, transform, label in MAPS:
            values = cells[col].to_numpy(float)
            g = roi_location.spatial_gradient(
                cells["overview_x_aligned"],
                cells["overview_y_aligned"],
                transform(values) if transform else values,
                n_perm=n_perm,
            )
            g["magnitude"] = g["magnitude"] / pixel_size * 100
            rows.append(
                dict(
                    session=session,
                    map=label,
                    x0=cells["overview_x_aligned"].mean(),
                    y0=cells["overview_y_aligned"].mean(),
                    **g,
                )
            )
    return pd.DataFrame(rows)


def plot_gradients_on_overview(
    fig,
    rects,
    overview_df,
    neurons_df_sig,
    xlims,
    ylims,
    pixel_size,
    fontsize_dict,
    arrow_px=40,
    alpha=0.05,
    nonsig_color="lightgrey",
    nonsig_alpha=0.7,
):
    """One overview map per gradient, with each session's arrow starting at its FOV centre.

    Same frame as Supplementary Figure 9 B-D (aligned overview coordinates, both
    axes inverted, visual area contours). Neurons are drawn as faint grey dots.
    Arrows are coloured when the shuffle p-value is below `alpha`, light grey
    otherwise. Within a map, arrow length is proportional to the gradient
    magnitude, with the median significant session `arrow_px` overview pixels
    long; a scale arrow gives that length in map units per 100 um.
    """
    from v1_depth_map.figure_utils.v1_map import add_visual_area_contours

    units = {"Azimuth": "deg", "Elevation": "deg", "Depth": "log$_2$ units"}
    axes = []
    for rect, (_, _, label), color in zip(rects, MAPS, MAP_COLORS):
        ax = fig.add_axes(rect)
        ax.scatter(
            neurons_df_sig["overview_x_aligned"],
            neurons_df_sig["overview_y_aligned"],
            s=0.3,
            color="0.93",
            linewidth=0,
            rasterized=True,
        )
        df = overview_df[overview_df["map"] == label]
        sig = df["pval_perm"].to_numpy() < alpha
        magnitude = df["magnitude"].to_numpy()
        ref = np.median(magnitude[sig]) if sig.any() else np.median(magnitude)
        # map units per overview pixel -> arrow length in overview pixels
        scale = arrow_px / (ref * pixel_size / 100)
        dx = df["slope_x"].to_numpy() * scale
        dy = df["slope_y"].to_numpy() * scale
        x0 = df["x0"].to_numpy()
        y0 = df["y0"].to_numpy()
        for mask, c, a in [(~sig, nonsig_color, nonsig_alpha), (sig, color, 1)]:
            ax.quiver(
                x0[mask],
                y0[mask],
                dx[mask],
                dy[mask],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=c,
                alpha=a,
                width=0.008,
                headwidth=3,
                headlength=4,
                headaxislength=3.5,
                zorder=3,
            )
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        add_visual_area_contours(ax, mode="black", linewidth=0.8)
        ax.set_title(
            f"{label} ({sig.sum()}/{len(sig)} sig.)",
            fontsize=fontsize_dict["label"],
            pad=2,
        )
        ax.text(
            0.5,
            -0.02,
            f"Arrow: {ref:.2g} {units[label]} / 100 µm",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=fontsize_dict["tick"],
        )
        axes.append(ax)
    return axes
