"""Visual area contours and publication plotting helpers for Figure 5.

This module provides exact vector contours for V1 and higher visual areas
(LM, AL, RL, A, AM, PM) along with modular panel plotters that assemble
Figure 5 into a unified publication graphic (17.5 cm x 11.44 cm).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import seaborn as sns

from cottage_analysis.plotting.style import rect_cm, panel_letter

# Exact vector contours normalized to subplot axes [0, 1] x [0, 1]
CONTOURS_DATA = {
    "black_paths": [
        {
            "vertices": [
                [0.7127, 0.633],
                [0.679, 0.6879],
                [0.6188, 0.7889],
                [0.5585, 0.8181],
                [0.4846, 0.854],
                [0.4027, 0.7689],
                [0.2896, 0.591],
                [0.2143, 0.4728],
                [0.1156, 0.343],
                [0.0911, 0.2075],
                [0.0855, 0.1762],
                [0.0785, 0.1331],
                [0.1008, 0.109],
                [0.1233, 0.0847],
                [0.202, 0.0923],
                [0.2366, 0.0919],
                [0.3278, 0.0906],
                [0.419, 0.093],
                [0.51, 0.0973],
                [0.6073, 0.1018],
                [0.706, 0.109],
                [0.7978, 0.1396],
                [0.8362, 0.1524],
                [0.8748, 0.1706],
                [0.8974, 0.2022],
                [0.9264, 0.243],
                [0.9024, 0.298],
                [0.883, 0.3434],
                [0.7127, 0.633],
                [0.7127, 0.633],
            ],
            "codes": [
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                79,
                1,
            ],
        },
        {
            "vertices": [
                [0.8267, 0.9727],
                [0.8395, 0.7611],
                [0.8633, 0.5501],
                [0.8979, 0.3406],
            ],
            "codes": [1, 4, 4, 4],
        },
        {
            "vertices": [
                [0.6923, 0.7492],
                [0.7168, 0.7519],
                [0.7419, 0.7517],
                [0.7666, 0.7514],
                [0.7804, 0.7513],
                [0.7952, 0.7507],
                [0.8082, 0.747],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4],
        },
        {
            "vertices": [
                [0.0024, 0.2664],
                [0.0242, 0.2517],
                [0.0441, 0.2344],
                [0.0614, 0.215],
            ],
            "codes": [1, 4, 4, 4],
        },
        {
            "vertices": [
                [0.2088, 0.568],
                [0.174, 0.5683],
                [0.1391, 0.5686],
                [0.1042, 0.5689],
                [0.0893, 0.569],
                [0.0739, 0.569],
                [0.06, 0.5643],
                [0.0425, 0.5583],
                [0.0294, 0.5452],
                [0.0194, 0.5313],
                [0.0119, 0.5209],
                [0.0058, 0.5099],
                [0.0003, 0.4985],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        },
        {
            "vertices": [
                [0.2607, 0.6331],
                [0.2526, 0.6566],
                [0.2443, 0.68],
                [0.2361, 0.7035],
                [0.2282, 0.726],
                [0.2202, 0.7488],
                [0.2061, 0.7685],
                [0.192, 0.7882],
                [0.1709, 0.8048],
                [0.1458, 0.8087],
                [0.1153, 0.8135],
                [0.0843, 0.7988],
                [0.0633, 0.7777],
                [0.0423, 0.7566],
                [0.0294, 0.7299],
                [0.0169, 0.7035],
                [-0.0023, 0.6631],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2],
        },
        {
            "vertices": [
                [0.4161, 0.9999],
                [0.4009, 0.9912],
                [0.3892, 0.983],
                [0.3714, 0.9718],
                [0.297, 0.9251],
                [0.2273, 0.8729],
                [0.1511, 0.8292],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4],
        },
        {
            "vertices": [
                [0.514, 0.8715],
                [0.5205, 0.9122],
                [0.5213, 0.9535],
                [0.5162, 0.9943],
                [0.5159, 0.997],
                [0.5155, 0.9997],
                [0.5149, 1.0023],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4],
        },
        {
            "vertices": [
                [0.7804, 0.981],
                [0.718, 0.9318],
                [0.6516, 0.887],
                [0.582, 0.8471],
            ],
            "codes": [1, 4, 4, 4],
        },
    ],
    "white_paths": [
        {
            "vertices": [
                [0.7104, 0.6331],
                [0.6767, 0.688],
                [0.6166, 0.789],
                [0.5562, 0.8182],
                [0.4824, 0.8541],
                [0.4005, 0.769],
                [0.2873, 0.5911],
                [0.2121, 0.4728],
                [0.1135, 0.343],
                [0.089, 0.2075],
                [0.0834, 0.1762],
                [0.0764, 0.1331],
                [0.0987, 0.109],
                [0.1212, 0.0846],
                [0.1999, 0.0922],
                [0.2344, 0.0918],
                [0.3256, 0.0906],
                [0.4167, 0.0929],
                [0.5077, 0.0972],
                [0.605, 0.1018],
                [0.7037, 0.1089],
                [0.7954, 0.1395],
                [0.8338, 0.1523],
                [0.8725, 0.1706],
                [0.895, 0.2022],
                [0.924, 0.243],
                [0.9, 0.298],
                [0.8806, 0.3434],
                [0.7104, 0.6331],
                [0.7104, 0.6331],
            ],
            "codes": [
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                79,
                1,
            ],
        },
        {
            "vertices": [
                [0.8229, 0.9972],
                [0.8357, 0.772],
                [0.8608, 0.5475],
                [0.8981, 0.3247],
            ],
            "codes": [1, 4, 4, 4],
        },
        {
            "vertices": [
                [0.6628, 0.7448],
                [0.6956, 0.7522],
                [0.7303, 0.7519],
                [0.7643, 0.7516],
                [0.7842, 0.7513],
                [0.8065, 0.7501],
                [0.8222, 0.74],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4],
        },
        {
            "vertices": [
                [0.2327, 0.5678],
                [0.1887, 0.5682],
                [0.1447, 0.5685],
                [0.1008, 0.5689],
                [0.0856, 0.5691],
                [0.0701, 0.5691],
                [0.056, 0.564],
                [0.0383, 0.5575],
                [0.0251, 0.5434],
                [0.0149, 0.5284],
                [-0.0005, 0.5056],
                [-0.0102, 0.48],
                [-0.0198, 0.4547],
                [-0.0281, 0.4326],
                [-0.0364, 0.4106],
                [-0.0448, 0.3885],
                [-0.0545, 0.3627],
                [-0.0641, 0.3334],
                [-0.0506, 0.3092],
                [-0.0413, 0.2923],
                [-0.023, 0.2819],
                [-0.0064, 0.2709],
                [0.0226, 0.2518],
                [0.0485, 0.2286],
                [0.0702, 0.2025],
            ],
            "codes": [
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ],
        },
        {
            "vertices": [
                [0.2662, 0.6114],
                [0.2555, 0.6421],
                [0.2447, 0.6729],
                [0.2339, 0.7036],
                [0.2261, 0.7261],
                [0.218, 0.7489],
                [0.2039, 0.7686],
                [0.1899, 0.7883],
                [0.1687, 0.8049],
                [0.1436, 0.8088],
                [0.1131, 0.8136],
                [0.0821, 0.7989],
                [0.0611, 0.7778],
                [0.0402, 0.7567],
                [0.0273, 0.7299],
                [0.0148, 0.7036],
                [0.005, 0.6831],
                [-0.0047, 0.6625],
                [-0.0145, 0.6419],
                [-0.0196, 0.6313],
                [-0.0248, 0.6199],
                [-0.0231, 0.6084],
                [-0.0201, 0.5868],
                [0.0035, 0.5755],
                [0.027, 0.5691],
            ],
            "codes": [
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ],
        },
        {
            "vertices": [
                [0.4105, 0.998],
                [0.3985, 0.9908],
                [0.3881, 0.9836],
                [0.3733, 0.9742],
                [0.2973, 0.9262],
                [0.2263, 0.8723],
                [0.1475, 0.8285],
            ],
            "codes": [1, 4, 4, 4, 4, 4, 4],
        },
        {
            "vertices": [
                [0.5077, 0.8491],
                [0.5177, 0.8991],
                [0.5198, 0.9505],
                [0.5139, 1.0011],
            ],
            "codes": [1, 4, 4, 4],
        },
        {
            "vertices": [
                [0.799, 0.998],
                [0.7261, 0.9384],
                [0.6474, 0.885],
                [0.5642, 0.8385],
            ],
            "codes": [1, 4, 4, 4],
        },
    ],
}

PANEL_A_LABELS = [
    ("A", (0.572, 0.979)),
    ("RL", (0.316, 0.902)),
    ("AM", (0.672, 0.890)),
    ("PM", (0.708, 0.727)),
    ("AL", (0.081, 0.723)),
    ("LM", (0.022, 0.487)),
    ("V1", (0.221, 0.307)),
]

FIG_W = 17.50
FIG_H = 11.44

LAYOUT = {
    "letters": {
        "A": (0.15, 11.15),
        "B": (6.30, 11.15),
        "C": (9.90, 11.15),
        "D": (13.50, 11.15),
        "E": (0.15, 5.00),
        "F": (7.80, 5.00),
    },
    "panel_a": (0.74, 5.95, 4.84, 5.21),
    "bcd_top": [
        (6.63, 8.72, 2.27, 2.44),
        (10.24, 8.72, 2.27, 2.44),
        (13.85, 8.72, 2.27, 2.44),
    ],
    "bcd_bot": [
        (6.63, 5.95, 2.27, 2.44),
        (10.24, 5.95, 2.27, 2.44),
        (13.85, 5.95, 2.27, 2.44),
    ],
    "bcd_cbar": [
        (9.12, 6.20, 0.12, 1.22),
        (12.70, 6.20, 0.12, 1.22),
        (16.32, 6.20, 0.12, 1.22),
    ],
    "panel_e_main": (1.09, 0.82, 4.58, 3.49),
    "panel_e_polar": (4.65, 3.40, 1.48, 1.48),
    "panel_e_cbar": (5.97, 0.90, 0.18, 1.75),
    "panel_f_scatter": [
        (8.97, 3.39, 2.50, 2.00),
        (11.67, 3.39, 2.50, 2.00),
        (14.37, 3.39, 2.50, 2.00),
    ],
    "panel_f_prop": [
        (8.97, 0.90, 2.50, 2.00),
        (11.67, 0.90, 2.50, 2.00),
        (14.37, 0.90, 2.50, 2.00),
    ],
    "panel_f_cbar": (16.34, 0.90, 0.09, 0.68),
}


def add_visual_area_contours(ax, mode="black", linewidth=0.8):
    """Overlay dashed visual area boundaries on a subplot in axes coordinates.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        mode (str): 'white' (for Panel A widefield overview) or 'black' (for Panels B-D).
        linewidth (float): Stroke width of the dashed boundary.
    """
    paths_data = (
        CONTOURS_DATA["white_paths"]
        if mode == "white"
        else CONTOURS_DATA["black_paths"]
    )
    color = "white" if mode == "white" else "black"
    dash_pattern = (0, (2.5, 2.5))
    for p in paths_data:
        v = np.array(p["vertices"])
        c = p["codes"]
        patch = PathPatch(
            Path(v, c),
            transform=ax.transAxes,
            fill=False,
            edgecolor=color,
            linestyle=dash_pattern,
            linewidth=linewidth,
            zorder=10,
        )
        ax.add_patch(patch)


def add_anatomical_compass(ax, center=(0.072, 0.933), length=0.040, color="white"):
    """Add anatomical orientation compass (Anterior, Posterior, Medial, Lateral).

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        center (tuple): Center coordinate (x, y) in axes fractions.
        length (float): Arm length in axes fractions.
        color (str): Stroke and text color.
    """
    cx, cy = center
    # Horizontal arm (Lateral to Medial)
    ax.annotate(
        "",
        xy=(cx + length, cy),
        xytext=(cx - length, cy),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.6),
    )
    # Vertical arm (Posterior to Anterior)
    ax.annotate(
        "",
        xy=(cx, cy + length),
        xytext=(cx, cy - length),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="<->", color=color, lw=0.6),
    )
    # Text labels
    pad = 0.015
    ax.text(
        cx,
        cy + length + pad,
        "A",
        transform=ax.transAxes,
        color=color,
        fontsize=5,
        ha="center",
        va="bottom",
    )
    ax.text(
        cx,
        cy - length - pad,
        "P",
        transform=ax.transAxes,
        color=color,
        fontsize=5,
        ha="center",
        va="top",
    )
    ax.text(
        cx + length + pad,
        cy,
        "M",
        transform=ax.transAxes,
        color=color,
        fontsize=5,
        ha="left",
        va="center",
    )
    ax.text(
        cx - length - pad,
        cy,
        "L",
        transform=ax.transAxes,
        color=color,
        fontsize=5,
        ha="right",
        va="center",
    )


def plot_panel_a(fig, rect, overview_img, pixel_size=6.395):
    """Plot Panel A: Widefield calcium imaging overview map."""
    ax = fig.add_axes(rect)
    ax.imshow(overview_img)
    xlims = [50, overview_img.shape[1] - 50]
    ylims = [50, overview_img.shape[0] - 50]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])

    # Visual area boundaries (white dashed)
    add_visual_area_contours(ax, mode="white", linewidth=1.0)

    # Visual area text labels
    for name, (x_ax, y_ax) in PANEL_A_LABELS:
        ax.text(
            x_ax,
            y_ax,
            name,
            transform=ax.transAxes,
            color="white",
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=12,
        )

    # Scale bar of 1 mm
    scalebar_length_px = 1000.0 / pixel_size
    rect_sb = plt.Rectangle(
        (overview_img.shape[1] - 50 - 180, overview_img.shape[0] - 50 - 30),
        scalebar_length_px,
        scalebar_length_px * 0.05,
        color="white",
        zorder=12,
    )
    ax.add_patch(rect_sb)

    # Anatomical compass
    add_anatomical_compass(ax, center=(0.070, 0.880), length=0.032, color="white")
    return ax


def plot_panels_bcd(
    fig,
    rects_top,
    rects_bot,
    rects_cbar,
    neurons_df_sig,
    maps,
    maps_alpha,
    xrange,
    yrange,
    xlims,
    ylims,
):
    """Plot Panels B-D: Cortical distributions of azimuth, elevation, and preferred depth."""
    cols = ["rf_azi", "rf_ele", "log_preferred_depth"]
    cmaps = [cm.YlOrRd.reversed(), cm.YlOrRd.reversed(), cm.cool.reversed()]
    vranges = [(17.5, 117.5), (-37.5, 37.5), (np.log(0.2), np.log(2.0))]
    cbar_titles = [
        "RF\nazimuth\n(degrees)",
        "RF\nelevation\n(degrees)",
        "Preferred\nvirtual\ndepth (cm)",
    ]

    for icol, (col, cmap, vrange, cbar_title) in enumerate(
        zip(cols, cmaps, vranges, cbar_titles)
    ):
        # Top subplot: scatter
        ax_top = fig.add_axes(rects_top[icol])
        df_clip = neurons_df_sig.copy()
        df_clip.loc[df_clip[col] < vrange[0], col] = vrange[0]
        df_clip.loc[df_clip[col] > vrange[1], col] = vrange[1]

        sns.scatterplot(
            data=df_clip,
            x="overview_x_aligned",
            y="overview_y_aligned",
            hue=col,
            palette=cmap,
            s=1,
            alpha=0.3,
            linewidth=0,
            legend=False,
            ax=ax_top,
            vmin=vrange[0],
            vmax=vrange[1],
            rasterized=True,
        )
        ax_top.set_xlim(xlims)
        ax_top.set_ylim(ylims)
        ax_top.invert_yaxis()
        ax_top.invert_xaxis()
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_xlabel("")
        ax_top.set_ylabel("")
        add_visual_area_contours(ax_top, mode="black", linewidth=0.8)

        # Bottom subplot: smoothed map
        ax_bot = fig.add_axes(rects_bot[icol])
        alpha = maps_alpha[col]
        alpha = np.clip(alpha / alpha.max() * 5, 0, 1)
        ax_bot.imshow(
            maps[col],
            extent=(xrange[0], xrange[1], yrange[0], yrange[1]),
            origin="lower",
            alpha=alpha,
            cmap=cmap,
            vmin=vrange[0],
            vmax=vrange[1],
        )
        ax_bot.set_xlim(xlims)
        ax_bot.set_ylim(ylims)
        ax_bot.invert_yaxis()
        ax_bot.invert_xaxis()
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        ax_bot.set_xlabel("")
        ax_bot.set_ylabel("")
        add_visual_area_contours(ax_bot, mode="black", linewidth=0.8)

        # Colorbar
        ax_cbar = fig.add_axes(rects_cbar[icol])
        norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, cax=ax_cbar)
        ax_cbar.tick_params(labelsize=5, length=2, pad=1)
        if col == "log_preferred_depth":
            cbar.set_ticks(vrange)
            cbar.set_ticklabels(["<20", ">200"])
        elif col == "rf_ele":
            cbar.set_ticks(vrange)
            cbar.set_ticklabels(["−37.5", "37.5"])
        else:
            cbar.set_ticks(vrange)
            cbar.set_ticklabels(["17.5", "117.5"])
        cbar.ax.set_title(cbar_title, fontsize=5, ha="left", pad=3)


def plot_panel_e(
    fig, rect_main, rect_polar, rect_cbar, median_depth, coefs_corrected, im_extent
):
    """Plot Panel E: Preferred virtual depth map across visual space and gradient polar plot."""
    ax = fig.add_axes(rect_main)
    vrange = [np.log(0.2), np.log(2.0)]
    _im = ax.imshow(
        np.log(median_depth),
        extent=im_extent,
        origin="lower",
        cmap="cool_r",
        vmin=vrange[0],
        vmax=vrange[1],
    )
    ax.set_xlabel("RF azimuth (degrees)", fontsize=7)
    ax.set_ylabel("RF elevation (degrees)", fontsize=7)
    ax.set_xticks([30, 60, 90, 120])
    ax.set_yticks([-40, 0, 40])
    ax.tick_params(labelsize=5, length=2, direction="out")
    sns.despine(ax=ax, offset=2)

    # Colorbar
    cbar_ax = fig.add_axes(rect_cbar)
    norm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    sm = plt.cm.ScalarMappable(cmap="cool_r", norm=norm)
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(np.log([0.2, 0.4, 1.0, 2.0]))
    cbar.set_ticklabels(["<20", "40", "100", ">200"])
    cbar.ax.tick_params(labelsize=5, pad=1)
    cbar.ax.set_title("Preferred\nvirtual\ndepth (cm)", fontsize=5, ha="left", pad=3)

    # Polar gradient inset
    ax_polar = fig.add_axes(rect_polar, polar=True)
    angles = np.arctan2(coefs_corrected[:, 1], coefs_corrected[:, 0])
    ax_polar.hist(angles, bins=50, color="black", edgecolor="black")
    ax_polar.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax_polar.set_xticklabels(["Temporal", "Upper", "Nasal", "Lower"], fontsize=5)
    ax_polar.set_yticks([])
    ax_polar.tick_params(axis="x", pad=-1)


def plot_panel_f(
    fig,
    rects_scatter,
    rects_prop,
    rect_cbar,
    neurons_df_sig,
    im_extent,
    calculate_zz_func,
    norm_e,
    depth_col="preferred_depth_corrected",
):
    """Plot Panel F: Representation across near, mid, and far depth categories in visual space.

    Args:
        depth_col (str): Column holding the preferred depth in metres. Defaults to the
            eccentricity-corrected depth used in Figure 5; pass
            "preferred_depth_closedloop" for the uncorrected supplementary version.
    """
    ranges = [[-np.inf, 0.2], [0.2, 1.0], [1.0, np.inf]]
    titles = ["<20 cm", "20-100 cm", ">100 cm"]
    this_col = depth_col

    im_prop = None
    for i, (r, title) in enumerate(zip(ranges, titles)):
        # Scatter plot
        ax_scat = fig.add_axes(rects_scatter[i])
        idx = (neurons_df_sig[this_col].values > r[0]) & (
            neurons_df_sig[this_col].values <= r[1]
        )
        sns.scatterplot(
            data=neurons_df_sig[idx],
            x="rf_azi_jittered",
            y="rf_ele_jittered",
            hue=np.log(neurons_df_sig[idx][this_col]),
            palette="cool_r",
            s=1,
            alpha=0.2,
            linewidth=0,
            legend=False,
            ax=ax_scat,
            hue_norm=norm_e,
            rasterized=True,
        )
        ax_scat.set_xlim([15, 120])
        ax_scat.set_ylim([-40, 40])
        ax_scat.set_xticks([30, 60, 90, 120])
        ax_scat.set_yticks([-40, 0, 40])
        ax_scat.tick_params(labelsize=5, length=2, direction="out")
        sns.despine(ax=ax_scat, offset=2)
        ax_scat.set_xlabel("")
        ax_scat.set_xticklabels([])
        if i == 0:
            ax_scat.set_ylabel("RF elevation (degrees)", fontsize=7)
        else:
            ax_scat.set_ylabel("")
            ax_scat.set_yticklabels([])
        ax_scat.set_title(title, fontsize=7, pad=2)

        # Proportion heatmap
        ax_prop = fig.add_axes(rects_prop[i])
        zz = calculate_zz_func(idx)
        im_prop = ax_prop.imshow(
            zz,
            extent=im_extent,
            origin="lower",
            cmap="Reds",
            vmin=0,
            vmax=0.6,
        )
        ax_prop.set_xlim([15, 120])
        ax_prop.set_ylim([-40, 40])
        ax_prop.set_xticks([30, 60, 90, 120])
        ax_prop.set_yticks([-40, 0, 40])
        ax_prop.tick_params(labelsize=5, length=2, direction="out")
        sns.despine(ax=ax_prop, offset=2)
        ax_prop.set_ylabel("")
        if i > 0:
            ax_prop.set_yticklabels([])
        if i == 1:
            ax_prop.set_xlabel("RF azimuth (degrees)", fontsize=7)
        else:
            ax_prop.set_xlabel("")

    # Colorbar
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = plt.colorbar(im_prop, cax=cbar_ax, ticks=[0, 0.6])
    cbar.ax.tick_params(labelsize=5, pad=1)
    cbar.ax.set_title("Proportion\nof neurons", fontsize=5, ha="left", pad=3)


def assemble_figure_5(
    fig,
    overview_img,
    neurons_df_sig,
    maps,
    maps_alpha,
    xrange,
    yrange,
    xlims,
    ylims,
    median_depth,
    coefs_corrected,
    im_extent,
    calculate_zz_func,
    pixel_size=6.395,
    depth_col="preferred_depth_corrected",
):
    """Assemble unified Figure 5 canvas matching publication layout (Panels A-F)."""
    fig.patch.set_facecolor("white")

    # Panel letters
    for letter, (x, y) in LAYOUT["letters"].items():
        panel_letter(fig, letter, x, y)

    # Panel A: Overview map
    plot_panel_a(
        fig, rect_cm(fig, *LAYOUT["panel_a"]), overview_img, pixel_size=pixel_size
    )

    # Panels B-D: Cortical retinotopic distributions
    rects_top = [rect_cm(fig, *r) for r in LAYOUT["bcd_top"]]
    rects_bot = [rect_cm(fig, *r) for r in LAYOUT["bcd_bot"]]
    rects_cbar = [rect_cm(fig, *r) for r in LAYOUT["bcd_cbar"]]
    plot_panels_bcd(
        fig,
        rects_top,
        rects_bot,
        rects_cbar,
        neurons_df_sig,
        maps,
        maps_alpha,
        xrange,
        yrange,
        xlims,
        ylims,
    )

    # Panel E: Preferred virtual depth map across visual space
    plot_panel_e(
        fig,
        rect_cm(fig, *LAYOUT["panel_e_main"]),
        rect_cm(fig, *LAYOUT["panel_e_polar"]),
        rect_cm(fig, *LAYOUT["panel_e_cbar"]),
        median_depth,
        coefs_corrected,
        im_extent,
    )

    # Panel F: Representation across depth categories
    rects_scat = [rect_cm(fig, *r) for r in LAYOUT["panel_f_scatter"]]
    rects_prop = [rect_cm(fig, *r) for r in LAYOUT["panel_f_prop"]]
    norm_e = plt.Normalize(vmin=np.log(0.2), vmax=np.log(2.0))
    plot_panel_f(
        fig,
        rects_scat,
        rects_prop,
        rect_cm(fig, *LAYOUT["panel_f_cbar"]),
        neurons_df_sig,
        im_extent,
        calculate_zz_func,
        norm_e,
        depth_col=depth_col,
    )
