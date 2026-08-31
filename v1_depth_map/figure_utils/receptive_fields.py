"""Drawing helpers for the 3D receptive-field panel of the RF figure.

The multi-depth receptive field of a neuron is a volume in visual space
(virtual depth x elevation x azimuth). matplotlib has no volume renderer, so
the volume is drawn as a stack of closely spaced marching-cubes isosurfaces
whose colour and opacity ramp from a pale outer envelope to a dark core, which
reads as one continuous cloud. Marginal projections are added on the floor and
the two back walls of the cube.

Typical use, once per example neuron (see `figures/figure_receptive_fields.ipynb`)::

    from v1_depth_map.figure_utils import receptive_fields as rf_utils

    ax = fig.add_subplot(111, projection="3d")
    for roi, cmap_name, color in zip(rois, cmaps, colors):
        volume = ...  # (ndepths, n_ele, n_azi), rectified, cropped in azimuth
        rf_utils.add_rf_isosurfaces(ax, volume, cmap_name)
        peak = rf_utils.add_rf_peak_marker(ax, volume, color)
        rf_utils.add_rf_marginals(ax, volume, color, peak[0])
    rf_utils.style_rf_3d_axes(ax, depths_arr)

The displayed window is set by `AZI_LIM`, `DEPTH_LIM` and `ELE_LIM`. Every
function takes them as keyword arguments.

Note that `AZI_LIM` is a *crop* of the coefficient grid, not a rescaling: the
volume handed to these functions must already be sliced to that window
(stimulus frames are `AZI_RESOLUTION` degrees per pixel starting at 0 degrees
azimuth, so `slice(azi_min // AZI_RESOLUTION, azi_max // AZI_RESOLUTION)`).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, zoom
from skimage import measure

# Displayed window of the cube: azimuth (deg), depth (index), elevation (deg)
AZI_LIM = (20, 120)
DEPTH_LIM = (0, 7)
ELE_LIM = (-35, 35)
AZI_RESOLUTION = 5  # degrees per stimulus-frame pixel

# Isosurface stack: fraction-of-peak range, number of shells, opacity and
# colormap position of the outermost shell / core. ISO_RANGE stops below 1
# because no surface exists exactly at the peak.
ISO_RANGE = (0.50, 0.98)
N_ISO = 12
ISO_ALPHA = (0.05, 0.50)
CMAP_RANGE = (0.25, 0.95)

# Marginal projections, kept faint so the volumes stay dominant
CONTOUR_FRACS = [0.40, 0.60, 0.80, 0.90]  # fraction of the marginal's peak
CONTOUR_LW = [0.5, 0.7, 0.9, 1.1]
MARGINAL_FILL_ALPHA = 0.05
MARGINAL_LINE_ALPHA = 0.3

# Smoothing / upsampling of the volume before marching cubes. The zoom factors
# set the triangle count, and so the weight of the vector output.
SMOOTH_SIGMA = (0.55, 0.75, 0.75)
VOLUME_ZOOM = (4, 3, 3)
MARGINAL_SMOOTH_SIGMA = 0.7
MARGINAL_ZOOM = (4, 4)

FONTSIZE_DICT = {"title": 7, "label": 7, "tick": 6, "legend": 5}


def add_rf_isosurfaces(
    ax,
    volume,
    cmap_name,
    light_source=None,
    azi_lim=AZI_LIM,
    depth_lim=DEPTH_LIM,
    ele_lim=ELE_LIM,
    iso_range=ISO_RANGE,
    n_iso=N_ISO,
    iso_alpha=ISO_ALPHA,
    cmap_range=CMAP_RANGE,
    rasterized=True,
):
    """Draw one neuron's RF volume as nested isosurfaces.

    Args:
        ax (Axes3D): 3D axes to draw into.
        volume (np.ndarray): Rectified RF volume, (ndepths, n_ele, n_azi),
            already cropped to `azi_lim`.
        cmap_name (str): Sequential colormap, e.g. "Purples". Shells are
            coloured along `cmap_range` of it, from outer envelope to core.
        light_source (LightSource, optional): Shading source. Defaults to
            LightSource(azdeg=135, altdeg=45).
        azi_lim, depth_lim, ele_lim (tuple): Displayed window of the cube.
        iso_range (tuple): Fraction of peak of the outermost shell / the core.
        n_iso (int): Number of shells. More is smoother but heavier.
        iso_alpha (tuple): Opacity of the outermost shell / the core.
        cmap_range (tuple): Colormap position of the outermost shell / the core.
        rasterized (bool): Rasterise the meshes so that saved SVG/PDF keeps
            axes, contours and text as vectors while the volumes become an
            image (set the `dpi` of `savefig` to control their resolution).
    """
    if light_source is None:
        light_source = LightSource(azdeg=135, altdeg=45)
    smooth = zoom(
        gaussian_filter(volume, sigma=list(SMOOTH_SIGMA)), VOLUME_ZOOM, order=3
    )
    vmax = smooth.max()
    nz, ny, nx = smooth.shape
    spacing = (
        (depth_lim[1] - depth_lim[0]) / (nz - 1),
        (ele_lim[1] - ele_lim[0]) / (ny - 1),
        (azi_lim[1] - azi_lim[0]) / (nx - 1),
    )
    cmap = plt.get_cmap(cmap_name)
    for i_iso, frac in enumerate(np.linspace(*iso_range, n_iso)):
        shade = i_iso / max(n_iso - 1, 1)  # 0 = outer shell, 1 = core
        rgb = np.array(
            cmap(cmap_range[0] + (cmap_range[1] - cmap_range[0]) * shade)[:3]
        )
        alpha = iso_alpha[0] + (iso_alpha[1] - iso_alpha[0]) * shade**1.5
        try:
            verts, faces, _, _ = measure.marching_cubes(
                smooth, level=frac * vmax, spacing=spacing
            )
        except Exception:
            # no surface at this level (e.g. a nearly flat volume)
            continue
        # X=Azimuth, Y=Depth, Z=Elevation
        mesh_verts = np.column_stack(
            [
                verts[:, 2] + azi_lim[0],
                verts[:, 0] + depth_lim[0],
                verts[:, 1] + ele_lim[0],
            ]
        )
        mesh = Poly3DCollection(mesh_verts[faces], alpha=alpha)
        normals = np.cross(
            mesh_verts[faces[:, 1]] - mesh_verts[faces[:, 0]],
            mesh_verts[faces[:, 2]] - mesh_verts[faces[:, 0]],
        )
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
        intensity = light_source.shade_normals(normals, fraction=0.7)
        mesh.set_facecolors(
            [(*np.clip(rgb * (0.65 + 0.35 * i), 0, 1), alpha) for i in intensity]
        )
        mesh.set_edgecolor("none")
        mesh.set_rasterized(rasterized)
        ax.add_collection3d(mesh)


def add_rf_peak_marker(
    ax,
    volume,
    color,
    azi_lim=AZI_LIM,
    depth_lim=DEPTH_LIM,
    ele_lim=ELE_LIM,
    markersize=16,
    facecolor="k",
    edgecolor="white",
):
    """Mark the peak of the volume and drop dashed lines to the cube walls.

    Args:
        ax (Axes3D): 3D axes to draw into.
        volume (np.ndarray): Rectified RF volume, (ndepths, n_ele, n_azi).
        color: Colour of the drop lines (usually the neuron's colour).
        azi_lim, depth_lim, ele_lim (tuple): Displayed window of the cube.
        markersize (float): Marker area, as matplotlib's `s`.
        facecolor, edgecolor: Marker face and surround.

    Returns:
        tuple: The peak index into `volume`, (idepth, i_ele, i_azi). Pass its
            first element to `add_rf_marginals` as `peak_idepth`.

    Note:
        mplot3d depth-sorts artists, so a marker inside a dense volume is drawn
        over by the front shells. Lower `iso_alpha[1]` or raise `iso_range[0]`
        in `add_rf_isosurfaces` to see it through the cloud (or create the axes
        with `computed_zorder=False`, at the cost of correct occlusion between
        neurons).
    """
    peak = np.unravel_index(np.argmax(volume), volume.shape)
    azi = np.linspace(*azi_lim, volume.shape[2])[peak[2]]
    ele = np.linspace(*ele_lim, volume.shape[1])[peak[1]]
    depth = peak[0]
    ax.scatter(
        [azi],
        [depth],
        [ele],
        color=facecolor,
        s=markersize,
        edgecolor=edgecolor,
        linewidth=0.8,
        zorder=20,
    )
    for xs, ys, zs in (
        ([azi, azi], [depth, depth], [ele, ele_lim[0]]),
        ([azi, azi], [depth, depth_lim[1]], [ele, ele]),
        ([azi, azi_lim[0]], [depth, depth], [ele, ele]),
    ):
        ax.plot(
            xs, ys, zs, color=color, linestyle=(0, (2, 2)), linewidth=0.8, alpha=0.55
        )
    return peak


def add_rf_marginals(
    ax,
    volume,
    color,
    peak_idepth,
    azi_lim=AZI_LIM,
    depth_lim=DEPTH_LIM,
    ele_lim=ELE_LIM,
    contour_fracs=CONTOUR_FRACS,
    contour_lw=CONTOUR_LW,
    fill_alpha=MARGINAL_FILL_ALPHA,
    line_alpha=MARGINAL_LINE_ALPHA,
):
    """Project the volume onto the floor and the two back walls of the cube.

    The floor shows azimuth x depth and the left wall depth x elevation (both
    maximum projections); the back wall shows azimuth x elevation of the slice
    at the peak depth.

    Args:
        ax (Axes3D): 3D axes to draw into.
        volume (np.ndarray): Rectified RF volume, (ndepths, n_ele, n_azi).
        color: Contour colour (usually the neuron's colour).
        peak_idepth (int): Depth index of the slice shown on the back wall,
            as returned by `add_rf_peak_marker`.
        azi_lim, depth_lim, ele_lim (tuple): Displayed window of the cube.
        contour_fracs (list): Contour levels, as fractions of each marginal's
            peak. The lowest one also bounds the filled region.
        contour_lw (list): Line width per level.
        fill_alpha, line_alpha (float): Opacity of the fill / the lines.
    """
    fill, line = [to_rgba(color, fill_alpha)], [to_rgba(color, line_alpha)]
    styles = ["-"] * len(contour_fracs)

    def fill_kwargs(marginal, **kwargs):
        return dict(
            levels=[contour_fracs[0] * marginal.max(), marginal.max()],
            colors=fill,
            **kwargs,
        )

    def line_kwargs(marginal, **kwargs):
        return dict(
            levels=[f * marginal.max() for f in contour_fracs],
            colors=line,
            linewidths=contour_lw,
            linestyles=styles,
            **kwargs,
        )

    def smooth_marginal(marginal):
        return zoom(
            gaussian_filter(marginal, MARGINAL_SMOOTH_SIGMA), MARGINAL_ZOOM, order=3
        )

    # Floor (Z = ele_min): azimuth x depth
    da = smooth_marginal(volume.max(axis=1))
    d_grid, a_grid = np.meshgrid(
        np.linspace(*depth_lim, da.shape[0]),
        np.linspace(*azi_lim, da.shape[1]),
        indexing="ij",
    )
    floor = dict(zdir="z", offset=ele_lim[0])
    ax.contourf(a_grid, d_grid, da, **fill_kwargs(da, **floor))
    ax.contour(a_grid, d_grid, da, **line_kwargs(da, **floor))

    # Back wall (Y = depth max): azimuth x elevation at the peak depth
    ea = smooth_marginal(volume[peak_idepth])
    a_grid3, e_grid3 = np.meshgrid(
        np.linspace(*azi_lim, ea.shape[1]),
        np.linspace(*ele_lim, ea.shape[0]),
        indexing="xy",
    )
    back = dict(zdir="y", offset=depth_lim[1])
    ax.contourf(a_grid3, ea, e_grid3, **fill_kwargs(ea, **back))
    ax.contour(a_grid3, ea, e_grid3, **line_kwargs(ea, **back))

    # Left wall (X = azi_min): depth x elevation
    de = smooth_marginal(volume.max(axis=2))
    d_grid2, e_grid2 = np.meshgrid(
        np.linspace(*depth_lim, de.shape[0]),
        np.linspace(*ele_lim, de.shape[1]),
        indexing="ij",
    )
    left = dict(zdir="x", offset=azi_lim[0])
    ax.contourf(de, d_grid2, e_grid2, **fill_kwargs(de, **left))
    ax.contour(de, d_grid2, e_grid2, **line_kwargs(de, **left))


def style_rf_3d_axes(
    ax,
    depths_arr,
    fontsize_dict=None,
    azi_lim=AZI_LIM,
    depth_lim=DEPTH_LIM,
    ele_lim=ELE_LIM,
    elev=28,
    azim=-50,
    box_aspect=(1.0, 1.0, 0.8),
):
    """Style the cube: white panes, ticks in cm and degrees, view angle.

    Args:
        ax (Axes3D): 3D axes to style.
        depths_arr (np.ndarray): Virtual depths in metres, used for the depth
            tick labels (converted to cm).
        fontsize_dict (dict, optional): Font sizes; defaults to FONTSIZE_DICT.
        azi_lim, depth_lim, ele_lim (tuple): Displayed window of the cube.
        elev, azim (float): View angle, matching the original plotly camera
            eye=(1.75, 1.75, 1.75).
        box_aspect (tuple): Relative length of the three axes.
    """
    fontsize_dict = fontsize_dict or FONTSIZE_DICT
    ax.set_facecolor("white")
    ax.set_box_aspect(box_aspect)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor((0.985, 0.985, 0.99, 0.9))
        pane.set_edgecolor((0.80, 0.80, 0.84, 1.0))
    ax.set_xlim(*azi_lim)
    ax.set_ylim(*depth_lim)
    ax.set_zlim(*ele_lim)
    azi_ticks = [t for t in [0, 30, 60, 90, 120] if azi_lim[0] <= t <= azi_lim[1]]
    ax.set_xticks(azi_ticks)
    ax.set_xticklabels([f"{t}°" for t in azi_ticks], fontsize=fontsize_dict["tick"])
    ax.set_yticks(np.arange(len(depths_arr)))
    ax.set_yticklabels(
        [f"{int(d * 100)}" for d in depths_arr], fontsize=fontsize_dict["tick"]
    )
    ax.set_zticks([-15, 0, 15])
    ax.set_zticklabels(["-15°", "0°", "+15°"], fontsize=fontsize_dict["tick"])
    ax.set_xlabel("Azimuth (degrees)", fontsize=fontsize_dict["label"], labelpad=3)
    ax.set_ylabel("Depth (cm)", fontsize=fontsize_dict["label"], labelpad=3)
    ax.set_zlabel("Elevation (degrees)", fontsize=fontsize_dict["label"], labelpad=3)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.90, 0.5)
        axis._axinfo["grid"]["linewidth"] = 0.4
    ax.view_init(elev=elev, azim=azim)
