"""Exploratory drifting-grating plots of the reviewer analysis (`sftf_exploratory_analysis.ipynb`).

Example neurons in the layout of the thesis example figure (`yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb`, cell 4),
on `ast_neuropil=False` traces and the cross-validated `neurons_df_sftf.pickle`, one band per neuron:
- depth tuning (trial means and Gaussian fit) and running speed / optic flow tuning matrix
  (`rsof_plots.plot_RS_OF_matrix`), from the closed-loop depth protocol;
- mean response to each SF x TF for the 8 directions, around a polar plot of the maximum per direction
  (`grating_plots.plot_sftf_tuning`);
- the same for the fitted model, on the same colour scale (`grating_plots.plot_sftf_fit`).

Also: SF x TF responses per running-speed bin, TF curves per speed bin and population SF / TF maps by preferred
depth.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cottage_analysis.analysis import gratings
from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import depth_selectivity_plots, grating_plots, rsof_plots
from v1_depth_map.figure_utils.sftf import FILTER_TRACES, PROJECT, load_depth_trials


def load_example_session(
    flexilims_session, session, photodiode_protocol=5, use_col="dffs"
):
    """Trial tables and fits of one grating session.

    `use_col` ("dffs" or "spks") is the response of the grating trials and fit
    (`neurons_df_sftf.pickle` or `neurons_df_sftf_spks.pickle`); the depth protocol is dF/F.

    Returns:
        dict: `trials_depth` (closed-loop depth protocol, from `spheres.sync_all_recordings`;
            open-loop "Playback" recordings dropped), `trials_sftf` (grating trials with one
            integer column per ROI, from `gratings.analyze_grating_responses`), `neurons_df` and
            `neurons_df_sftf`.
    """
    trials_depth = load_depth_trials(flexilims_session, session, photodiode_protocol)
    trials_sftf, _ = gratings.analyze_grating_responses(
        project=PROJECT,
        session=session,
        filter_datasets=FILTER_TRACES,
        photodiode_protocol=photodiode_protocol,
        protocol_base="SFTF",
        use_col=use_col,
    )
    sftf_fname = (
        "neurons_df_sftf.pickle"
        if use_col == "dffs"
        else f"neurons_df_sftf_{use_col}.pickle"
    )
    folder = pipeline_utils.create_neurons_ds(
        session_name=session,
        flexilims_session=flexilims_session,
        project=PROJECT,
        conflicts="skip",
    ).path_full.parent
    return dict(
        trials_depth=trials_depth,
        trials_sftf=trials_sftf,
        neurons_df=pd.read_pickle(folder / "neurons_df.pickle"),
        neurons_df_sftf=pd.read_pickle(folder / sftf_fname),
    )


def plot_example_neuron(
    fig,
    data,
    roi,
    top,
    fontsize_dict,
    ylim_precision_base=1,
    ylim_precision=1,
    rsof_log_range=None,
    rsof_vmax=None,
):
    """Plot one example neuron in a band of `fig` whose top is at `top` (figure fraction).

    The band is ~0.65 high: tuning curves on top, raw (left) and fitted (right) SF/TF
    responses below. Layout and settings are those of the thesis figure.
    `rsof_log_range` is passed to `rsof_plots.plot_RS_OF_matrix` as `log_range` (its
    default if None). `rsof_vmax="max"` sets the matrix colour scale to the max of the plotted
    matrix (the default of `plot_RS_OF_matrix` leaves out the last RS column and OF row).
    """
    ax = fig.add_axes([0.05, top, 0.15, 0.1])
    depth_selectivity_plots.plot_depth_tuning_curve(
        neurons_df=data["neurons_df"],
        trials_df=data["trials_depth"],
        roi=roi,
        rs_thr=None,
        plot_fit=True,
        plot_smooth=False,
        linewidth=2,
        linecolor="royalblue",
        closed_loop=1,
        fontsize_dict=fontsize_dict,
        markersize=10,
        markeredgecolor="w",
        ylim_precision_base=ylim_precision_base,
        ylim_precision=ylim_precision,
        ax=ax,
    )
    rsof_kwargs = {} if rsof_log_range is None else dict(log_range=rsof_log_range)
    if rsof_vmax == "max":
        tmp_fig, tmp_ax = plt.subplots()
        *_, matrix = rsof_plots.plot_RS_OF_matrix(
            trials_df=data["trials_depth"],
            roi=roi,
            is_closed_loop=1,
            cbar_width=None,
            ax=tmp_ax,
            return_matrix=True,
            **rsof_kwargs,
        )
        plt.close(tmp_fig)
        rsof_vmax = np.nanmax(matrix)
    ax = fig.add_axes([0.3, top - 0.02, 0.15, 0.14])
    rsof_plots.plot_RS_OF_matrix(
        trials_df=data["trials_depth"],
        roi=roi,
        is_closed_loop=1,
        vmin=0,
        vmax=rsof_vmax,
        **rsof_kwargs,
        fontsize_dict=fontsize_dict,
        ax=ax,
    )
    fig.axes[-1].yaxis.set_major_formatter(
        "{x:.2g}"
    )  # colorbar added by plot_RS_OF_matrix

    trials_sftf = data["trials_sftf"]
    sf = np.sort(trials_sftf.SpatialFrequency.unique())
    tf = np.sort(trials_sftf.TemporalFrequency.unique())
    grid = dict(
        plot_grid=False,
        plot_axes=True,
        fig=fig,
        plot_y=top - 0.2,
        plot_width=0.4,
        plot_height=0.4,
    )
    # shared colour / polar scale: max of the condition means and of the fit peak, rounded up
    # (the default of plot_sftf_tuning rounds the data max to 0.1, which can clip both)
    data_max = (
        trials_sftf.groupby(["Angle", "SpatialFrequency", "TemporalFrequency"])[roi]
        .mean()
        .max()
    )
    fit = data["neurons_df_sftf"].iloc[roi]
    angles = np.linspace(0, 2 * np.pi, 100)
    fit_max = grating_tuning(
        (np.full_like(angles, fit.sf0), np.full_like(angles, fit.tf0), angles),
        *GratingParams(**{k: fit[k] for k in GratingParams._fields}),
        min_sigma=0.25,
    ).max()
    vmax = max(data_max, fit_max)
    decimals = 1 - int(np.floor(np.log10(vmax)))  # 2 significant digits
    vmax = np.ceil(vmax * 10**decimals) / 10**decimals
    grating_plots.plot_sftf_tuning(
        trials_df=trials_sftf,
        roi=roi,
        plot_x=0.05,
        fontsize_dict=fontsize_dict,
        colorbar=False,
        vmax=vmax,
        **grid,
    )
    grating_plots.plot_sftf_fit(
        neurons_df=data["neurons_df_sftf"],
        roi=roi,
        sf_range=[np.log(sf[0]), np.log(sf[-1])],
        tf_range=[np.log(tf[0]), np.log(tf[-1])],
        sf_ticks=sf,
        tf_ticks=tf,
        min_sigma=0.25,
        plot_x=0.55,
        fontsize_dict=fontsize_dict,
        colorbar=True,
        vmax=vmax,
        **grid,
    )


def plot_sftf_by_speed(
    fig, binned, fontsize_dict, title=None, width=None, gap=0.04, bottom=0.1
):
    """Raw SF x TF responses per direction (no fit) for each speed bin, side by side, on a
    common colour scale. `binned` is the output of `speed_binned_responses`.

    Panel sizes are figure fractions; the height is scaled by the figure aspect ratio so the
    3 x 3 grids are square. The grids' bottom row starts at `bottom`. By default the grids
    share 90% of the figure width.
    """
    if width is None:
        width = (0.9 - gap * (len(binned) - 1)) / len(binned)
    fig_w, fig_h = fig.get_size_inches()
    height = width * fig_w / fig_h
    top = bottom + 2 * height / 3  # bottom edge of the top row
    vmax = max(
        np.nanmax(
            df.groupby(["SpatialFrequency", "TemporalFrequency", "Angle"])[0].mean()
        )
        for _, df in binned
    )
    vmax = np.round(vmax, 2)
    for i, (label, df) in enumerate(binned):
        x = 0.05 + i * (width + gap)
        grating_plots.plot_sftf_tuning(
            trials_df=df,
            roi=0,
            plot_grid=False,
            plot_axes=True,
            fig=fig,
            plot_x=x,
            plot_y=top,
            plot_width=width,
            plot_height=height,
            fontsize_dict=fontsize_dict,
            colorbar=i == len(binned) - 1,
            vmax=vmax,
        )
        fig.text(
            x + width / 2,
            top + height / 3,
            label,
            ha="center",
            va="bottom",
            fontsize=fontsize_dict["title"],
        )
    if title is not None:
        fig.text(
            0.02,
            top + height / 3 + 0.08,
            title,
            ha="left",
            va="bottom",
            fontsize=fontsize_dict["title"],
            fontweight="bold",
        )


def plot_tf_curves(ax, curves, cmap="viridis", fontsize_dict=None, ylabel="ΔF/F"):
    """One TF (or SF) curve (mean +- SEM across pairs) per speed bin, slow to fast.

    The frequency is the `TemporalFrequency` or `SpatialFrequency` column of `curves`.
    """
    freq = "TemporalFrequency" if "TemporalFrequency" in curves else "SpatialFrequency"
    fontsize_dict = fontsize_dict or {"label": 6, "tick": 5, "legend": 5}
    labels = list(dict.fromkeys(curves["bin"]))
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(labels)))
    for label, color in zip(labels, colors):
        c = curves[curves["bin"] == label]
        ax.errorbar(
            c[freq],
            c["mean"],
            yerr=c["sem"],
            color=color,
            lw=1,
            marker="o",
            ms=2,
            capsize=0,
            label=label,
        )
    ax.set_xscale("log")
    values = np.sort(curves[freq].unique())
    ax.set_xticks(values, [f"{v:g}" for v in values])
    ax.minorticks_off()
    ax.set_xlabel(
        "Temporal frequency (Hz)"
        if freq == "TemporalFrequency"
        else "Spatial frequency (cpd)",
        fontsize=fontsize_dict["label"],
    )
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(labelsize=fontsize_dict["tick"])
    ax.spines[["top", "right"]].set_visible(False)


def extract_sftf_population_data(cells_df, cache_dir):
    """Extract condition-mean SFTF response tensors for all cells in cells_df.

    Args:
        cells_df (pd.DataFrame): Table of cells (must contain 'session', 'roi',
            'preferred_depth', and optionally 'pref_dir', 'pref_sf', 'pref_tf').
        cache_dir (Path): Directory containing cached `{session}_sftf_trials.pickle` files.

    Returns:
        dict:
            'cells': pd.DataFrame with cell metadata and computed arrays,
            'angle_range': array of stimulus drift angles (deg),
            'sf_range': array of spatial frequencies (cpd),
            'tf_range': array of temporal frequencies (Hz),
    """
    sessions = cells_df["session"].unique()
    all_cells = []
    angle_range = None
    sf_range = None
    tf_range = None

    for session in sessions:
        cache_path = cache_dir / f"{session}_sftf_trials.pickle"
        if not cache_path.exists():
            continue
        d = pd.read_pickle(cache_path)
        trials_df = d["trials"]
        dff = d["dff"]  # (n_trials, n_rois)
        sess_cells = cells_df[cells_df.session == session]

        if angle_range is None:
            angle_range = np.sort(trials_df.Angle.unique())
            sf_range = np.sort(trials_df.SpatialFrequency.unique())
            tf_range = np.sort(trials_df.TemporalFrequency.unique())

        sess_rois = sess_cells["roi"].values
        sess_dff = dff[:, sess_rois]  # (n_trials, n_cells_in_session)

        # Precompute mean responses per (angle, sf, tf) condition
        mat = np.zeros((len(angle_range), len(sf_range), len(tf_range), len(sess_rois)))
        for ia, a in enumerate(angle_range):
            for isf, s in enumerate(sf_range):
                for itf, t in enumerate(tf_range):
                    mask = (
                        (trials_df.Angle == a)
                        & (trials_df.SpatialFrequency == s)
                        & (trials_df.TemporalFrequency == t)
                    )
                    mat[ia, isf, itf, :] = sess_dff[mask.values].mean(axis=0)

        for i_c, (_, row) in enumerate(sess_cells.iterrows()):
            cell_mat = mat[:, :, :, i_c]  # (8, 6, 6)
            dir_prof = cell_mat.max(axis=(1, 2))
            pref_dir_idx = int(np.argmax(dir_prof))
            aligned_mat = np.roll(cell_mat, -pref_dir_idx, axis=0)
            peak = float(cell_mat.max())
            if peak > 0:
                all_cells.append(
                    {
                        "session": session,
                        "roi": int(row["roi"]),
                        "preferred_depth": float(row["preferred_depth"]),
                        "pref_dir": float(
                            row.get("pref_dir", angle_range[pref_dir_idx])
                        ),
                        "pref_sf": float(row.get("pref_sf", np.nan)),
                        "pref_tf": float(row.get("pref_tf", np.nan)),
                        "peak_response": peak,
                        "mat_raw": cell_mat,
                        "mat_aligned": aligned_mat,
                        "mat_norm": cell_mat / peak,
                        "mat_norm_aligned": aligned_mat / peak,
                    }
                )

    return {
        "cells": pd.DataFrame(all_cells),
        "angle_range": angle_range,
        "sf_range": sf_range,
        "tf_range": tf_range,
    }


def bin_cells_by_depth(cells_df, strategy="tertiles", edges=None):
    """Assign depth tuning bins (near, mid, far, or near, far) to cells.

    Args:
        cells_df (pd.DataFrame): Cells table with 'preferred_depth' (in metres).
        strategy (str): 'tertiles' (equal count, 3 bins),
            'halves' or 'median' (equal count cut in middle, 2 bins),
            'fixed' (<20cm, 20-100cm, >100cm, 3 bins),
            'fixed_2bins' or 'fixed_100' (<=100cm, >100cm, 2 bins),
            or 'custom' (requires `edges`).
        edges (list, optional): Bin edges in metres for 'custom'.

    Returns:
        pd.DataFrame: Copy of cells_df with 'depth_bin' and 'bin_label' columns,
        dict: {bin_name: label_string_with_n}.
    """
    df = cells_df.copy()
    if strategy in ("halves", "median", "2bins"):
        q50 = df.preferred_depth.median()
        bin_edges = [-np.inf, q50, np.inf]
        labels = ["near", "far"]
        df["depth_bin"] = pd.cut(df.preferred_depth, bin_edges, labels=labels)
        counts = df["depth_bin"].value_counts()
        label_dict = {
            "near": f"Near (depth ≤ {q50 * 100:.0f} cm)\nn = {counts.get('near', 0)}",
            "far": f"Far (depth > {q50 * 100:.0f} cm)\nn = {counts.get('far', 0)}",
        }
    elif strategy in ("fixed_2bins", "fixed_100"):
        bin_edges = [-np.inf, 1.00, np.inf]
        labels = ["near", "far"]
        df["depth_bin"] = pd.cut(df.preferred_depth, bin_edges, labels=labels)
        counts = df["depth_bin"].value_counts()
        label_dict = {
            "near": f"Near (depth ≤ 100 cm)\nn = {counts.get('near', 0)}",
            "far": f"Far (depth > 100 cm)\nn = {counts.get('far', 0)}",
        }
    elif strategy == "tertiles":
        q33, q67 = df.preferred_depth.quantile([1 / 3, 2 / 3])
        bin_edges = [-np.inf, q33, q67, np.inf]
        labels = ["near", "mid", "far"]
        df["depth_bin"] = pd.cut(df.preferred_depth, bin_edges, labels=labels)
        counts = df["depth_bin"].value_counts()
        label_dict = {
            "near": f"Near (depth ≤ {q33 * 100:.0f} cm)\nn = {counts.get('near', 0)}",
            "mid": f"Mid ({q33 * 100:.0f}–{q67 * 100:.0f} cm)\nn = {counts.get('mid', 0)}",
            "far": f"Far (depth > {q67 * 100:.0f} cm)\nn = {counts.get('far', 0)}",
        }
    elif strategy == "fixed":
        bin_edges = [-np.inf, 0.20, 1.00, np.inf]
        labels = ["near", "mid", "far"]
        df["depth_bin"] = pd.cut(df.preferred_depth, bin_edges, labels=labels)
        counts = df["depth_bin"].value_counts()
        label_dict = {
            "near": f"Near (< 20 cm)\nn = {counts.get('near', 0)}",
            "mid": f"Mid (20–100 cm)\nn = {counts.get('mid', 0)}",
            "far": f"Far (> 100 cm)\nn = {counts.get('far', 0)}",
        }
    elif strategy == "custom":
        if edges is None:
            raise ValueError("edges must be specified for 'custom' strategy")
        if len(edges) == 3:
            labels = ["near", "far"]
        elif len(edges) == 4:
            labels = ["near", "mid", "far"]
        else:
            labels = [f"bin_{i}" for i in range(len(edges) - 1)]
        df["depth_bin"] = pd.cut(df.preferred_depth, edges, labels=labels)
        counts = df["depth_bin"].value_counts()
        label_dict = {b: f"{b.capitalize()}\nn = {counts.get(b, 0)}" for b in labels}
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    df["bin_label"] = df["depth_bin"].map(label_dict)
    return df, label_dict


def plot_sftf_population_tuning(
    pop_data,
    strategy="tertiles",
    metric="raw",
    align_direction=False,
    vmax=None,
    vmin=0,
    fontsize_dict=None,
    fig=None,
):
    """Plot average SFTF data for 3 populations (Near, Mid, Far) like single-neuron examples.

    Layout: 3 columns (Near, Mid, Far). In each column, an 8-direction grid of
    Spatial Frequency x Temporal Frequency heatmaps surrounds a central polar
    direction tuning curve.

    Args:
        pop_data (dict): Output from `extract_sftf_population_data`.
        strategy (str): 'tertiles' or 'fixed'.
        metric (str): 'raw' (mean ΔF/F) or 'norm' (peak-normalized per cell).
        align_direction (bool): If True, circularly shift directions so each cell's
            preferred direction is at Δ0°. If False, retain absolute physical drift directions.
        vmax (float, optional): Maximum value for color and polar scale. If None, auto-calculated.
        vmin (float, optional): Minimum value for color and polar scale. If None, auto-calculated.
        fontsize_dict (dict, optional): Font size dictionary.
        fig (matplotlib.Figure, optional): Existing figure to plot onto.

    Returns:
        matplotlib.Figure: The plotted figure.
    """
    fontsize_dict = fontsize_dict or {"title": 7, "label": 7, "tick": 5, "legend": 5}
    cells_df, label_dict = bin_cells_by_depth(pop_data["cells"], strategy=strategy)
    angle_range = pop_data["angle_range"]
    sf_range = pop_data["sf_range"]
    tf_range = pop_data["tf_range"]

    mat_key = (
        ("mat_norm_aligned" if align_direction else "mat_norm")
        if metric == "norm"
        else ("mat_aligned" if align_direction else "mat_raw")
    )

    pop_bins = list(label_dict.keys())
    n_pops = len(pop_bins)
    pop_matrices = {}
    for b in pop_bins:
        sub = cells_df[cells_df.depth_bin == b]
        if len(sub) > 0:
            pop_matrices[b] = np.stack(sub[mat_key].values).mean(axis=0)
        else:
            pop_matrices[b] = np.zeros((len(angle_range), len(sf_range), len(tf_range)))

    if vmax is None:
        data_max = max(m.max() for m in pop_matrices.values())
        decimals = 1 - int(np.floor(np.log10(data_max))) if data_max > 0 else 1
        vmax = np.ceil(data_max * 10**decimals) / 10**decimals
    if vmin is None:
        data_min = min(m.min() for m in pop_matrices.values())
        decimals = 1 - int(np.floor(np.log10(data_min))) if data_min > 0 else 1
        vmin = np.floor(data_min * 10**decimals) / 10**decimals

    if fig is None:
        fig_w_cm = 13.0 if n_pops == 2 else 18.0
        fig_h_cm = 6.2
        fig = plt.figure(figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54))

    grid_map = {
        (0, 0): 135,
        (0, 1): 90,
        (0, 2): 45,
        (1, 0): 180,
        (1, 2): 0,
        (2, 0): 225,
        (2, 1): 270,
        (2, 2): 315,
    }

    w_block = 0.38 if n_pops == 2 else 0.26
    h_block = 0.76
    y_base = 0.14

    for i_pop, b in enumerate(pop_bins):
        mats = pop_matrices[b]
        x_base = (0.07 + i_pop * 0.44) if n_pops == 2 else (0.05 + i_pop * 0.30)
        sub_w = w_block / 3.25
        sub_h = h_block / 3.25
        gap_x = (w_block - 3 * sub_w) / 2
        gap_y = (h_block - 3 * sub_h) / 2

        # Title
        fig.text(
            x_base + w_block / 2,
            y_base + h_block + 0.03,
            label_dict[b],
            ha="center",
            va="bottom",
            fontsize=fontsize_dict["title"],
            fontweight="bold",
        )

        # Center Polar Tuning
        polar_ax = fig.add_axes(
            [
                x_base + sub_w + gap_x + 0.001,
                y_base + sub_h + gap_y + 0.001,
                sub_w * 0.84,
                sub_h * 0.84,
            ],
            projection="polar",
        )
        dir_tuning = np.nanmax(mats, axis=(1, 2))
        dir_tuning = np.append(dir_tuning, dir_tuning[0])
        dirs = np.deg2rad(np.append(angle_range, angle_range[0]))

        polar_ax.plot(dirs, dir_tuning, color="k", linewidth=0.7)
        polar_ax.fill_between(dirs, dir_tuning, color="#4A90E2", alpha=0.7)
        polar_ax.set_ylim(0, vmax)
        polar_ax.set_yticks([0, vmax])
        polar_ax.set_yticklabels(["0", f"{vmax:g}"], fontsize=fontsize_dict["tick"] - 1)
        polar_ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        tick_fmt = "Δ{d}°" if align_direction else "{d}°"
        polar_ax.set_xticklabels(
            [tick_fmt.format(d=int(d)) for d in [0, 45, 90, 135, 180, 225, 270, 315]],
            fontsize=fontsize_dict["tick"] - 2,
        )
        polar_ax.tick_params(axis="x", pad=-3)

        # 8 Directional heatmaps
        im_last = None
        for row in range(3):
            for col in range(3):
                if (row, col) == (1, 1):
                    continue
                ang = grid_map[(row, col)]
                ang_idx = np.where(angle_range == ang)[0][0]

                ax_x = x_base + col * (sub_w + gap_x)
                ax_y = y_base + (2 - row) * (sub_h + gap_y)
                ax_dir = fig.add_axes([ax_x, ax_y, sub_w, sub_h])

                im_last = ax_dir.imshow(
                    mats[ang_idx].T,
                    origin="lower",
                    vmax=vmax,
                    vmin=vmin,
                    cmap="magma",
                    aspect="auto",
                )

                if col == 0:
                    ax_dir.set_yticks(range(len(tf_range)))
                    ax_dir.set_yticklabels(
                        [f"{t:g}" for t in tf_range], fontsize=fontsize_dict["tick"] - 2
                    )
                    if row == 1:
                        ax_dir.set_ylabel(
                            "TF (Hz)", fontsize=fontsize_dict["label"] - 1
                        )
                else:
                    ax_dir.set_yticks([])
                    ax_dir.set_yticklabels([])

                if row == 2:
                    ax_dir.set_xticks(range(len(sf_range)))
                    ax_dir.set_xticklabels(
                        [f"{s:g}" for s in sf_range],
                        fontsize=fontsize_dict["tick"] - 2,
                        rotation=90,
                    )
                    if col == 1:
                        ax_dir.set_xlabel(
                            "SF (cpd)", fontsize=fontsize_dict["label"] - 1
                        )
                else:
                    ax_dir.set_xticks([])
                    ax_dir.set_xticklabels([])

    cbar_x = 0.93 if n_pops == 2 else 0.94
    cbar_ax = fig.add_axes([cbar_x, y_base + 0.05, 0.012, h_block - 0.1])
    cb = fig.colorbar(im_last, cax=cbar_ax)
    cb_label = "Normalized Response (peak = 1)" if metric == "norm" else "Mean ΔF/F"
    cb.set_label(cb_label, fontsize=fontsize_dict["label"])
    cb.ax.tick_params(labelsize=fontsize_dict["tick"])

    return fig


def plot_sftf_population_marginals(
    pop_data,
    strategy="tertiles",
    metric="raw",
    fontsize_dict=None,
    fig=None,
    cmap="cool_r",
):
    """Plot collapsed 2D SF x TF heatmaps and 1D marginal tuning curves for populations.

    Args:
        pop_data (dict): Output from `extract_sftf_population_data`.
        strategy (str): 'tertiles', 'halves'/'median', 'fixed', 'fixed_2bins'.
        metric (str): 'raw' or 'norm'.
        fontsize_dict (dict, optional): Font size dictionary.
        fig (matplotlib.Figure, optional): Matplotlib figure.
        cmap (str or Colormap, optional): Colormap for depth populations. Defaults to "cool_r".

    Returns:
        matplotlib.Figure: The plotted figure.
    """
    fontsize_dict = fontsize_dict or {"title": 7, "label": 7, "tick": 5, "legend": 5}
    cells_df, label_dict = bin_cells_by_depth(pop_data["cells"], strategy=strategy)
    sf_range = pop_data["sf_range"]
    tf_range = pop_data["tf_range"]

    mat_key = "mat_norm" if metric == "norm" else "mat_raw"
    pop_bins = list(label_dict.keys())
    n_pops = len(pop_bins)

    if isinstance(cmap, str):
        if cmap == "cool":
            cmap_obj = plt.get_cmap("cool_r")
        else:
            cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    if n_pops == 2:
        colors = {
            "near": cmap_obj(0.0),
            "far": cmap_obj(1.0),
        }
    else:
        colors = {
            "near": cmap_obj(0.0),
            "mid": cmap_obj(0.5),
            "far": cmap_obj(1.0),
        }

    if fig is None:
        fig_w_cm = 18.0
        fig_h_cm = 5.0
        fig = plt.figure(figsize=(fig_w_cm / 2.54, fig_h_cm / 2.54))

    # Layout dimensions
    if n_pops == 2:
        w_hm = 0.14
        gap_hm = 0.025
        x_hm_start = 0.06
        cbar_x = x_hm_start + 2 * (w_hm + gap_hm) + 0.005
        ax_sf_x = 0.51
        ax_tf_x = 0.76
        w_line = 0.19
    else:
        w_hm = 0.135
        gap_hm = 0.015
        x_hm_start = 0.05
        cbar_x = x_hm_start + 3 * (w_hm + gap_hm) + 0.005
        ax_sf_x = 0.62
        ax_tf_x = 0.83
        w_line = 0.15

    h_hm = 0.65
    y_hm = 0.22

    vmin = np.inf
    vmax = 0
    pop_2d = {}
    for b in pop_bins:
        sub = cells_df[cells_df.depth_bin == b]
        mats = np.stack(sub[mat_key].values).mean(axis=0)  # (8, 6, 6)
        sftf_2d = mats.mean(axis=0)  # (6, 6)
        pop_2d[b] = sftf_2d
        vmax = max(vmax, sftf_2d.max())
        vmin = min(vmin, sftf_2d.min())

    decimals = 1 - int(np.floor(np.log10(vmax))) if vmax > 0 else 1
    vmax = np.ceil(vmax * 10**decimals) / 10**decimals
    vmin = np.floor(vmin * 10**decimals) / 10**decimals

    im = None
    for i, b in enumerate(pop_bins):
        ax = fig.add_axes([x_hm_start + i * (w_hm + gap_hm), y_hm, w_hm, h_hm])
        im = ax.imshow(
            pop_2d[b].T,
            origin="lower",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(label_dict[b], fontsize=fontsize_dict["title"], fontweight="bold")
        ax.set_xticks(range(len(sf_range)))
        ax.set_xticklabels(
            [f"{s:g}" for s in sf_range],
            fontsize=fontsize_dict["tick"],
            rotation=90,
        )
        ax.set_xlabel("SF (cpd)", fontsize=fontsize_dict["label"])
        if i == 0:
            ax.set_yticks(range(len(tf_range)))
            ax.set_yticklabels(
                [f"{t:g}" for t in tf_range], fontsize=fontsize_dict["tick"]
            )
            ax.set_ylabel("TF (Hz)", fontsize=fontsize_dict["label"])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])

    cbar_ax = fig.add_axes([cbar_x, y_hm, 0.012, h_hm])
    cb = fig.colorbar(im, cax=cbar_ax)
    cb.set_label(
        "Normalized" if metric == "norm" else "Mean ΔF/F",
        fontsize=fontsize_dict["label"],
    )
    cb.ax.tick_params(labelsize=fontsize_dict["tick"])

    # 1D SF tuning curve
    ax_sf = fig.add_axes([ax_sf_x, y_hm, w_line, h_hm])
    ax_tf = fig.add_axes([ax_tf_x, y_hm, w_line, h_hm])

    for b in pop_bins:
        sub = cells_df[cells_df.depth_bin == b]
        # per cell 2D SFTF then marginal
        cell_mats = np.stack(sub[mat_key].values)  # (N, 8, 6, 6)
        cell_2d = cell_mats.mean(axis=1)  # (N, 6, 6)
        cell_sf = cell_2d.mean(axis=2)  # (N, 6)
        cell_tf = cell_2d.mean(axis=1)  # (N, 6)

        mean_sf, sem_sf = cell_sf.mean(axis=0), cell_sf.std(axis=0) / np.sqrt(len(sub))
        mean_tf, sem_tf = cell_tf.mean(axis=0), cell_tf.std(axis=0) / np.sqrt(len(sub))

        ax_sf.errorbar(
            sf_range,
            mean_sf,
            yerr=sem_sf,
            color=colors[b],
            lw=1.2,
            marker="o",
            ms=2.5,
            label=b.capitalize(),
        )
        ax_tf.errorbar(
            tf_range,
            mean_tf,
            yerr=sem_tf,
            color=colors[b],
            lw=1.2,
            marker="o",
            ms=2.5,
            label=b.capitalize(),
        )

    ax_sf.set_xscale("log")
    ax_sf.set_xticks(sf_range, [f"{s:g}" for s in sf_range])
    ax_sf.set_xlabel("SF (cpd)", fontsize=fontsize_dict["label"])
    ax_sf.set_ylabel(
        "Normalized" if metric == "norm" else "Mean ΔF/F",
        fontsize=fontsize_dict["label"],
    )
    ax_sf.tick_params(labelsize=fontsize_dict["tick"], axis="x", rotation=90)
    ax_sf.spines[["top", "right"]].set_visible(False)
    ax_sf.legend(fontsize=fontsize_dict["legend"], frameon=False)

    ax_tf.set_xscale("log")
    ax_tf.set_xticks(tf_range, [f"{t:g}" for t in tf_range])
    ax_tf.set_xlabel("TF (Hz)", fontsize=fontsize_dict["label"])
    ax_tf.tick_params(labelsize=fontsize_dict["tick"], axis="x", rotation=90)
    ax_tf.spines[["top", "right"]].set_visible(False)

    return fig
