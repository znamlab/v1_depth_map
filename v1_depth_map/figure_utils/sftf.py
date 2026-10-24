"""Example neurons for the reviewer grating figure: depth, optic flow and SF/TF tuning.

Port of the example-neuron figure of the thesis (`yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb`,
cell 4) to the current data: `ast_neuropil=False` traces and the cross-validated
`neurons_df_sftf.pickle`. One row per neuron:

- depth tuning (trial means and Gaussian fit) and running speed / optic flow tuning
  matrix (`rsof_plots.plot_RS_OF_matrix`), from the closed-loop depth protocol;
- mean response to each SF x TF for the 8 directions, around a polar plot of the maximum
  per direction (`grating_plots.plot_sftf_tuning`);
- the same for the fitted model, on the same colour scale (`grating_plots.plot_sftf_fit`).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cottage_analysis.analysis import gratings, spheres
from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import depth_selectivity_plots, grating_plots, rsof_plots

PROJECT = "hey2_3d-vision_foodres_20220101"
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}


def load_depth_trials(flexilims_session, session, photodiode_protocol=5):
    """Closed-loop trials of the depth protocol (`spheres.sync_all_recordings`, dF/F).

    Open-loop "Playback" recordings are dropped (the plots also select closed_loop == 1;
    dropping them here keeps every use closed loop).
    """
    _, trials_depth = spheres.sync_all_recordings(
        session_name=session,
        flexilims_session=flexilims_session,
        project=PROJECT,
        filter_datasets=FILTER_TRACES,
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    return trials_depth[trials_depth.closed_loop == 1].reset_index(drop=True)


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


def speed_binned_responses(
    X, response, log_rs_median, n_bins=3, edges_cm_s=None, recording=None
):
    """Frame responses of one ROI split into running-speed bins of equal frame counts.

    Args:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - `log_rs_median`, as returned
            by `precompute_data.sftf_running_models.session_frames` (running frames only).
        response (n_frames,): dF/F of the ROI on the same frames.
        log_rs_median (float): median log RS (m/s) subtracted from X[3].
        n_bins (int): number of speed bins (quantiles of log RS), if `edges_cm_s` is None.
        edges_cm_s (array, optional): bin edges in cm/s; frames outside are dropped.
        recording (array, optional): recording name per frame to identify trial boundaries.

    Returns:
        list of (label, DataFrame): one table per bin with `SpatialFrequency`,
            `TemporalFrequency`, `Angle` (deg), `trial` ID and the response in column 0.
    """
    log_rs = X[3] + log_rs_median
    if edges_cm_s is None:
        edges = np.quantile(log_rs, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, log_rs, side="right") - 1, 0, n_bins - 1)
    else:
        edges = np.log(np.asarray(edges_cm_s) / 100)
        n_bins = len(edges) - 1
        bins = np.searchsorted(edges, log_rs, side="right") - 1  # -1 or n_bins: outside
    stim = X[:3].round(4)
    if recording is not None:
        new_trial = np.r_[
            True,
            (np.diff(stim, axis=1) != 0).any(axis=0)
            | (recording[1:] != recording[:-1]),
        ]
    else:
        new_trial = np.r_[True, (np.diff(stim, axis=1) != 0).any(axis=0)]
    trial = np.cumsum(new_trial)
    frames = pd.DataFrame(
        {
            "SpatialFrequency": np.exp(X[0]).round(4),
            "TemporalFrequency": np.exp(X[1]).round(4),
            "Angle": np.rad2deg(X[2]).round().astype(int),
            "trial": trial,
            0: response,
        }
    )
    cm_s = np.exp(edges) * 100
    return [
        (
            f"{cm_s[i]:.1f}-{cm_s[i + 1]:.1f} cm/s",
            frames[bins == i].reset_index(drop=True),
        )
        for i in range(n_bins)
    ]


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


FREQ_COLS = {"tf": "TemporalFrequency", "sf": "SpatialFrequency"}


def sftf_model_kept_pairs(
    cell_fits, sf_values, tf_values, angle_values, variable="tf", threshold=0.5
):
    """Determine kept (SF, Angle) or (TF, Angle) pairs using the fitted SFTF parametric model.

    Args:
        cell_fits: list/array of 10 parameters, Series, or DataFrame containing GratingParams fields.
        sf_values: tested spatial frequencies in cpd.
        tf_values: tested temporal frequencies in Hz.
        angle_values: tested drift directions in degrees or radians.
        variable: "tf" (kept SF x Angle pairs for TF tuning) or "sf" (kept TF x Angle pairs for SF tuning).
        threshold: fraction of peak pair response (default 0.5).

    Returns:
        kept: boolean array of shape (n_sf, n_ang, n_cells) if variable == "tf",
              or (n_tf, n_ang, n_cells) if variable == "sf".
    """
    from cottage_analysis.analysis.fit_gaussian_blob import (
        GratingParams,
        grating_tuning,
    )

    param_names = list(GratingParams._fields)
    if isinstance(cell_fits, (list, tuple, np.ndarray)):
        popt_mat = np.atleast_2d(cell_fits)
    elif isinstance(cell_fits, pd.Series):
        prefix = (
            "fit_"
            if f"fit_{param_names[0]}" in cell_fits
            else ("base_" if f"base_{param_names[0]}" in cell_fits else "")
        )
        popt_mat = np.array([[cell_fits[f"{prefix}{p}"] for p in param_names]])
    elif isinstance(cell_fits, pd.DataFrame):
        prefix = (
            "fit_"
            if f"fit_{param_names[0]}" in cell_fits.columns
            else ("base_" if f"base_{param_names[0]}" in cell_fits.columns else "")
        )
        popt_mat = cell_fits[[f"{prefix}{p}" for p in param_names]].to_numpy()
    else:
        raise ValueError("Unsupported cell_fits type")

    sf_v = np.asarray(sf_values, float)
    tf_v = np.asarray(tf_values, float)
    ang_v = np.asarray(angle_values, float)
    if ang_v.max() > 2 * np.pi:
        ang_v = np.deg2rad(ang_v)

    SF, TF, ANG = np.meshgrid(np.log(sf_v), np.log(tf_v), ang_v, indexing="ij")
    stim = np.stack([SF.ravel(), TF.ravel(), ANG.ravel()])  # (3, 288)

    n_cells = len(popt_mat)
    kept_list = []
    for i in range(n_cells):
        popt = popt_mat[i]
        offset = popt[6]
        pred = (
            grating_tuning(stim, *popt, min_sigma=0.25).reshape(
                len(sf_v), len(tf_v), len(ang_v)
            )
            - offset
        )
        if variable == "tf":
            pair_resp = pred.mean(axis=1)  # (n_sf, n_ang)
        else:
            pair_resp = pred.mean(axis=0)  # (n_tf, n_ang)
        peak = pair_resp.max()
        kept_list.append(
            pair_resp >= threshold * peak
            if peak > 0
            else np.zeros_like(pair_resp, bool)
        )
    return np.stack(kept_list, axis=-1)


def tf_curves_by_speed(
    binned, threshold=0.5, variable="tf", kept_pairs=None, popt=None
):
    """TF (or SF) tuning per speed bin, over the SF (TF) / direction pairs that drive the cell.

    The pairs kept are either specified directly via `kept_pairs`, computed from the SFTF model
    parameters `popt`, or chosen as those whose mean response (over TF and all frames of all bins)
    is at least `threshold` x that of the best pair. For each bin, the curve is the mean over kept
    pairs of the condition mean at each TF, with the SEM across pairs. With `variable="sf"`, the
    roles of SF and TF are swapped.

    Args:
        binned: output of `speed_binned_responses`.
        threshold (float): fraction of the best pair's response.
        variable (str): "tf" or "sf", the frequency along the curve.
        kept_pairs (list, optional): list of kept (other frequency, Angle) pairs.
        popt (array/Series, optional): fitted SFTF GratingParams to derive kept pairs from the model.

    Returns:
        (DataFrame, list): one row per bin and frequency (`bin`, `TemporalFrequency` or
            `SpatialFrequency`, `mean`, `sem`, `n_pairs`), and the kept (other frequency, Angle)
            pairs.
    """
    freq = FREQ_COLS[variable]
    other = FREQ_COLS["sf" if variable == "tf" else "tf"]
    keys = [other, "Angle"]
    if kept_pairs is not None:
        kept = kept_pairs
    elif popt is not None:
        all_df = pd.concat([df for _, df in binned])
        sf_v = np.sort(all_df["SpatialFrequency"].unique())
        tf_v = np.sort(all_df["TemporalFrequency"].unique())
        ang_v = np.deg2rad(np.sort(all_df["Angle"].unique()))
        kept_mask = sftf_model_kept_pairs(
            popt, sf_v, tf_v, ang_v, variable=variable, threshold=threshold
        )[..., 0]
        if variable == "tf":
            kept = [
                (sf_v[i], int(np.rad2deg(ang_v[j]).round()))
                for i in range(len(sf_v))
                for j in range(len(ang_v))
                if kept_mask[i, j]
            ]
        else:
            kept = [
                (tf_v[i], int(np.rad2deg(ang_v[j]).round()))
                for i in range(len(tf_v))
                for j in range(len(ang_v))
                if kept_mask[i, j]
            ]
    else:
        pair_resp = pd.concat([df for _, df in binned]).groupby(keys)[0].mean()
        kept = list(pair_resp[pair_resp >= threshold * pair_resp.max()].index)
    rows = []
    for label, df in binned:
        sel = df[pd.MultiIndex.from_frame(df[keys]).isin(kept)]
        if "trial" in sel.columns:
            trial_resp = sel.groupby([freq, "trial"])[0].mean().reset_index()
            per_f = trial_resp.groupby(freq)[0].agg(["mean", "sem", "count"])
        else:
            cond = sel.groupby(keys + [freq])[0].mean()
            per_f = cond.groupby(freq).agg(["mean", "sem", "count"])
        rows.append(
            per_f.rename(
                columns={"count": "n_trials" if "trial" in sel.columns else "n_pairs"}
            )
            .assign(bin=label)
            .reset_index()
        )
    return pd.concat(rows, ignore_index=True), kept


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


def plot_tf_curves_with_models(
    ax,
    data_curves,
    model_curves_dict,
    cmap="viridis",
    fontsize_dict=None,
):
    """Plot data TF curves by speed bin with overlaid model predictions.

    Args:
        ax: matplotlib axes.
        data_curves: output of tf_curves_by_speed for real data.
        model_curves_dict: dict of {model_label: (curves_df, linestyle)}.
        cmap: colormap for speed bins.
        fontsize_dict: font sizes.
    """
    fontsize_dict = fontsize_dict or {"label": 6, "tick": 5, "legend": 5}
    labels = list(dict.fromkeys(data_curves["bin"]))
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(labels)))

    # Plot data points
    for label, color in zip(labels, colors):
        c = data_curves[data_curves["bin"] == label]
        ax.errorbar(
            c.TemporalFrequency,
            c["mean"],
            yerr=c["sem"],
            color=color,
            lw=1,
            marker="o",
            ms=2,
            capsize=0,
            label=label,
        )

    # Plot model curves
    for model_name, (m_curves, ls) in model_curves_dict.items():
        for label, color in zip(labels, colors):
            c = m_curves[m_curves["bin"] == label]
            ax.plot(
                c.TemporalFrequency,
                c["mean"],
                color=color,
                linestyle=ls,
                lw=1.2,
                alpha=0.8,
            )

    ax.set_xscale("log")
    tf = np.sort(data_curves.TemporalFrequency.unique())
    ax.set_xticks(tf, [f"{t:g}" for t in tf])
    ax.minorticks_off()
    ax.set_xlabel("Temporal frequency (Hz)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("ΔF/F", fontsize=fontsize_dict["label"])
    ax.tick_params(labelsize=fontsize_dict["tick"])
    ax.spines[["top", "right"]].set_visible(False)


def plot_model_selection_r2_gains(ax, summary_df, fontsize_dict=None):
    """Plot held-out test R² gain over base per model across 10 cells."""
    fontsize_dict = fontsize_dict or {"label": 6, "tick": 5, "legend": 5}
    models = [
        "gain",
        "offset",
        "gain_offset",
        "tf",
        "gain_offset_tf",
        "gain_offset_sf",
        "gain_offset_tf_sf",
        "gain_offset_speed",
    ]
    model_labels = [
        "gain",
        "offset",
        "gain+\noffset",
        "tf",
        "gain+off\n+tf",
        "gain+off\n+sf",
        "gain+off\n+tf+sf",
        "gain+off\n+speed",
    ]
    x = np.arange(len(models))

    # One thin line per cell
    for _, row in summary_df.iterrows():
        gains = [row[f"{m}_test_rsq"] - row["base_test_rsq"] for m in models]
        ax.plot(x, gains, color="gray", alpha=0.4, lw=1)

    # Median line
    median_gains = [
        (summary_df[f"{m}_test_rsq"] - summary_df["base_test_rsq"]).median()
        for m in models
    ]
    ax.plot(
        x,
        median_gains,
        color="royalblue",
        marker="o",
        ms=4,
        lw=2,
        label="Median",
    )
    ax.axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.5)

    ax.set_xticks(x, model_labels, fontsize=fontsize_dict["tick"])
    ax.set_ylabel("Held-out ΔR² vs base", fontsize=fontsize_dict["label"])
    ax.tick_params(labelsize=fontsize_dict["tick"])
    ax.spines[["top", "right"]].set_visible(False)


def plot_fitted_rs_params(axes, summary_df, fontsize_dict=None):
    """Plot fitted gain, offset, and TF shift slopes across 10 cells."""
    fontsize_dict = fontsize_dict or {"label": 6, "tick": 5, "legend": 5}

    # axes[0]: multiplicative gain g (gain_offset)
    # axes[1]: additive offset b (gain_offset)
    # axes[2]: TF shift a_tf (tf model vs gain_offset_tf)
    ax_g, ax_b, ax_tf = axes

    # Gain
    g_vals = summary_df["gain_offset_g"]
    ax_g.scatter(
        np.zeros_like(g_vals) + np.random.normal(0, 0.05, len(g_vals)),
        g_vals,
        color="royalblue",
        alpha=0.7,
        s=20,
    )
    ax_g.axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.5)
    ax_g.set_title(
        f"Gain slope g\n(med: {g_vals.median():.2f})", fontsize=fontsize_dict["label"]
    )
    ax_g.set_xticks([])
    ax_g.set_ylabel("d(log amp) / d(log RS)", fontsize=fontsize_dict["label"])
    ax_g.spines[["top", "right"]].set_visible(False)

    # Offset
    b_vals = summary_df["gain_offset_b"]
    ax_b.scatter(
        np.zeros_like(b_vals) + np.random.normal(0, 0.05, len(b_vals)),
        b_vals,
        color="crimson",
        alpha=0.7,
        s=20,
    )
    ax_b.axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.5)
    ax_b.set_title(
        f"Offset slope b\n(med: {b_vals.median():.3f})", fontsize=fontsize_dict["label"]
    )
    ax_b.set_xticks([])
    ax_b.set_ylabel("d(offset) / d(log RS)", fontsize=fontsize_dict["label"])
    ax_b.spines[["top", "right"]].set_visible(False)

    # TF shifts: tf alone vs gain_offset_tf
    atf_alone = summary_df["tf_a_tf"]
    atf_with_go = summary_df["gain_offset_tf_a_tf"]
    x1 = np.zeros(len(atf_alone))
    x2 = np.ones(len(atf_with_go))
    for v1, v2 in zip(atf_alone, atf_with_go):
        ax_tf.plot([0, 1], [v1, v2], color="gray", alpha=0.4, lw=1)
    ax_tf.scatter(x1, atf_alone, color="darkorange", alpha=0.7, s=20, label="TF alone")
    ax_tf.scatter(
        x2, atf_with_go, color="seagreen", alpha=0.7, s=20, label="TF + gain+off"
    )
    ax_tf.axhline(0, color="k", linestyle="--", lw=0.8, alpha=0.5)
    ax_tf.set_xticks(
        [0, 1], ["TF alone", "TF +\ngain+off"], fontsize=fontsize_dict["tick"]
    )
    ax_tf.set_ylabel("TF slope a_tf", fontsize=fontsize_dict["label"])
    ax_tf.set_title(
        f"a_tf comparison\n({atf_alone.median():.2f} → {atf_with_go.median():.2f})",
        fontsize=fontsize_dict["label"],
    )
    ax_tf.spines[["top", "right"]].set_visible(False)


def population_tf_curves(
    X,
    dff,
    log_rs_median,
    edges_cm_s,
    threshold=0.5,
    variable="tf",
    select=None,
    use=None,
    kept_pairs=None,
):
    """TF (or SF) tuning per speed bin for many cells at once (vectorised `tf_curves_by_speed`).

    For each cell, the SF / direction pairs whose mean response over all frames in the speed range is
    at least `threshold` x that of the best pair are kept; the curve of a speed bin is the mean over the
    kept pairs of the condition mean at each TF.

    Args:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - `log_rs_median`.
        dff (n_frames, n_cells): responses.
        log_rs_median (float): median log RS (m/s) subtracted from X[3].
        edges_cm_s (array): speed bin edges in cm/s; frames outside are dropped. None: one bin
            with all frames (e.g. still frames, whose X[3] is 0), labelled "all".
        threshold (float): fraction of the best pair's response.
        variable (str): "tf" for TF curves over SF / direction pairs, "sf" for SF curves over
            TF / direction pairs.
        select (n_frames,) bool, optional: frames used to choose the pairs (default: all).
        use (n_frames,) bool, optional: frames used for the curves (default: all), e.g. the other
            half of the trials for cross-validated curves.
        kept_pairs (array, optional): boolean array of kept pairs (n_other, n_ang, n_cells).
            If provided, `select` is not used for pair selection (e.g. model-based selection).

    Returns:
        curves (n_cells, n_bins, n_values): mean response (NaN where no kept pair has frames),
        values (n_values,): TF (Hz) or SF (cpd),
        labels (n_bins,): bin labels.
    """
    if variable == "sf":  # swap the roles of SF and TF
        X = X[[1, 0, 2, 3]]
    if edges_cm_s is None:
        n_bins = 1
        bins = np.zeros(X.shape[1], int)
    else:
        edges = np.log(np.asarray(edges_cm_s) / 100)
        n_bins = len(edges) - 1
        bins = np.searchsorted(edges, X[3] + log_rs_median, side="right") - 1
    ok = (bins >= 0) & (bins < n_bins)
    sf_v, i_sf = np.unique(X[0, ok].round(4), return_inverse=True)
    tf_v, i_tf = np.unique(X[1, ok].round(4), return_inverse=True)
    ang_v, i_ang = np.unique(X[2, ok].round(4), return_inverse=True)
    n_sf, n_tf, n_ang = len(sf_v), len(tf_v), len(ang_v)
    y = dff[ok]
    n_cells = y.shape[1]
    cond = ((bins[ok] * n_sf + i_sf) * n_ang + i_ang) * n_tf + i_tf
    n_cond = n_bins * n_sf * n_ang * n_tf

    def condition_sums(mask):
        # sums and counts per (bin, sf, angle, tf) over the frames in `mask`
        mask = np.ones(ok.sum(), bool) if mask is None else np.asarray(mask)[ok]
        sums = np.zeros((n_cond, n_cells))
        np.add.at(sums, cond[mask], y[mask])
        counts = np.bincount(cond[mask], minlength=n_cond).astype(float)
        return (
            sums.reshape(n_bins, n_sf, n_ang, n_tf, n_cells),
            counts.reshape(n_bins, n_sf, n_ang, n_tf),
        )

    # pair selection: from provided model `kept_pairs` or from `select` frames of the range
    if kept_pairs is not None:
        kept = kept_pairs
        sums, counts = condition_sums(use)
    else:
        sel_sums, sel_counts = condition_sums(select)
        pair_mean = (
            sel_sums.sum(axis=(0, 3)) / sel_counts.sum(axis=(0, 3))[..., None]
        )  # (sf, ang, cell)
        kept = pair_mean >= threshold * pair_mean.max(axis=(0, 1), keepdims=True)
        sums, counts = (sel_sums, sel_counts) if use is select else condition_sums(use)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond_mean = sums / counts[..., None]  # (bin, sf, ang, tf, cell)
    cond_mean = np.where(kept[None, :, :, None, :], cond_mean, np.nan)
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            curves = np.nanmean(cond_mean, axis=(1, 2))  # (bin, tf, cell)
    if edges_cm_s is None:
        labels = ["all"]
    else:
        cm_s = np.exp(edges) * 100
        labels = [f"{cm_s[i]:.1f}-{cm_s[i + 1]:.1f} cm/s" for i in range(n_bins)]
    # values rounded so that they print as the stimulus values (0.5, 1, ..., 16 Hz)
    return np.moveaxis(curves, -1, 0), np.exp(tf_v).round(3), labels


def tf_centre_of_mass(curves, tf):
    """Gain-invariant preferred TF: centre of mass in log2 TF of (curve - min of curve).

    Subtracting the minimum and normalising by the sum makes the value independent of a
    multiplicative gain (and of an additive offset) applied to the curve.

    Args:
        curves (..., n_tf): TF curves.
        tf (n_tf,): temporal frequencies (Hz).

    Returns:
        (...,) centre of mass in log2 Hz, NaN for flat curves.
    """
    w = curves - np.nanmin(curves, axis=-1, keepdims=True)
    w = np.where(np.isfinite(w), w, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (w * np.log2(tf)).sum(axis=-1) / w.sum(axis=-1)


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


# ==============================================================================
# Figure 3 SFTF Panel Helpers
# ==============================================================================

# running-speed bins (cm/s); the first starts at 5 cm/s, the still threshold
SPEED_EDGES_CM_S = [5, 11.31, 22.63, 45.25, 90.51]
TF_SELECT_THRESHOLD = 0.5
P_FREQ_TUNED = 0.05
MIN_CELLS_PER_POINT = 10
POP_FIT_MIN_SIGMA = 0.1
# version of the held-out speed curves in the summary cache: 2 = static curve as bin -1, halves by recording
SPEED_CURVES_VERSION = 2


def _freq_modulation_p(trials, resp, variable, threshold=TF_SELECT_THRESHOLD):
    """One-way ANOVA p-value of the TF (or SF) effect at kept SF (or TF) / direction pairs (raw responses)."""
    from scipy import stats

    freq = FREQ_COLS[variable]
    other = FREQ_COLS["sf" if variable == "tf" else "tf"]
    pair = trials.groupby([other, "Angle"]).ngroup().to_numpy()
    i_f = np.unique(trials[freq], return_inverse=True)[1]
    P = np.eye(pair.max() + 1)[pair]
    F = np.eye(i_f.max() + 1)[i_f]
    pair_mean = P.T @ resp / P.sum(axis=0)[:, None]
    kept = pair_mean >= threshold * pair_mean.max(axis=0)
    mask = kept[pair]
    yc = np.where(mask, resp, 0)
    n_f = F.T @ mask
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_f = (F.T @ yc) / n_f
        grand_mean = yc.sum(axis=0) / mask.sum(axis=0)
        ss_between = np.nansum(n_f * (mean_f - grand_mean) ** 2, axis=0)
        ss_within = np.nansum(np.where(mask, (resp - mean_f[i_f]) ** 2, 0), axis=0)
        df_between = (n_f > 0).sum(axis=0) - 1
        df_within = mask.sum(axis=0) - (n_f > 0).sum(axis=0)
        f_stat = (ss_between / df_between) / (ss_within / df_within)
    return stats.f.sf(f_stat, df_between, df_within)


def _frame_halves(recording, recordings):
    """Alternate presentations (0 / 1) within each SF x TF x direction condition.

    Each SFTF recording shows every condition once, so the k-th presentation of a condition is in
    the k-th recording: the half of a frame is the parity of its recording's rank in `recordings`
    (all SFTF recordings of the session, in time order). Unlike ranking the presentations found in
    the frames, this gives the same halves for any subset of the frames (running or still).
    """
    rank = {name: i for i, name in enumerate(sorted(recordings))}
    return np.array([rank[r] for r in recording]) % 2


def _align_to_peak(curves_sel, curves_out):
    """Peak frequency index and normalisation from curves_sel applied to curves_out."""
    mean_curve = np.nanmean(curves_sel, axis=1)
    peak = np.argmax(np.where(np.isnan(mean_curve), -np.inf, mean_curve), axis=1)
    peak_value = mean_curve[np.arange(len(peak)), peak]
    return peak, curves_out / peak_value[:, None, None]


def _bootstrap_spearman(x, y, sessions, n_boots=20000, seed=0):
    """Hierarchical bootstrap Spearman rho, 95% CI, and p-value over sessions.

    Mice are not resampled: with 2 mice, half the resamples would contain a single mouse.
    """
    import contextlib
    import io
    from cottage_analysis.analysis import common_utils

    df_boot = pd.DataFrame(
        dict(x=np.asarray(x), y=np.asarray(y), session=np.asarray(sessions))
    )
    df_boot = df_boot[np.isfinite(df_boot.x) & np.isfinite(df_boot.y)].reset_index(
        drop=True
    )
    np.random.seed(seed)
    with contextlib.redirect_stderr(io.StringIO()):
        rho, dist = common_utils.hierarchical_bootstrap_stats(
            df_boot,
            n_boots,
            ["x"],
            ["session"],
            ycol=["y"],
            correlation=True,
            plot=False,
        )
    return dict(
        n=len(df_boot),
        n_sessions=df_boot.session.nunique(),
        n_mice=df_boot.session.str.split("_").str[0].nunique(),
        rho=float(rho[0]),
        ci=np.percentile(dist[:, 0], [2.5, 97.5]),
        p=float(common_utils.calculate_pval_from_bootstrap(dist[:, 0], value=0)),
    )


def load_fig3_sftf_data(
    read_root,
    example_session="PZAH10.2d_S20230922",
    example_roi=558,
    dsi_thr=0.6,
    n_boots=20000,
    rebuild=False,
):
    """Load and compute all SFTF data required for the bottom rows of Figure 3."""
    import flexiznam as flz
    from cottage_analysis.analysis import common_utils

    sftf_root = read_root / "fig_reviewer_gratings"
    cache_dir = sftf_root / "cache"
    frames_dir = sftf_root / "running_models" / "frames"
    still_dir = sftf_root / "running_models" / "frames_still"

    cells = pd.read_csv(sftf_root / "grating_tuning_cells.csv")
    tuned = cells[cells.sftf_tuned & cells.depth_tuned].reset_index(drop=True)
    tuned_other = cells[cells.sftf_tuned & ~cells.depth_tuned].reset_index(drop=True)
    dir_tuned = tuned[tuned.fit_dsi > dsi_thr].reset_index(drop=True)
    dir_tuned_other = tuned_other[tuned_other.fit_dsi > dsi_thr].reset_index(drop=True)

    # 1. Example neuron closed-loop depth tuning (cached per single ROI to stay < 100 KB)
    ex_depth_cache = (
        cache_dir / f"{example_session}_roi{example_roi}_depth_example.pickle"
    )
    if ex_depth_cache.exists() and not rebuild:
        ex_depth = pd.read_pickle(ex_depth_cache)
    else:
        flm_sftf = flz.get_flexilims_session(PROJECT)
        trials_depth = load_depth_trials(flm_sftf, example_session)
        folder = pipeline_utils.create_neurons_ds(
            session_name=example_session,
            flexilims_session=flm_sftf,
            project=PROJECT,
            conflicts="skip",
        ).path_full.parent
        ndf_full = pd.read_pickle(folder / "neurons_df.pickle")
        # Keep only column 0 for the example ROI so the cache file is ~50 KB
        td_slim = trials_depth[["depth", "closed_loop", "RS_stim"]].copy()
        td_slim["dff_stim"] = [
            arr[:, [example_roi]] for arr in trials_depth["dff_stim"]
        ]
        ndf_slim = ndf_full.iloc[[example_roi]].reset_index(drop=True)
        ex_depth = dict(neurons_df=ndf_slim, trials_depth=td_slim, roi=0)
        pd.to_pickle(ex_depth, ex_depth_cache)

    # 2. Example neuron SFTF 3x3 condition means
    trials_data = pd.read_pickle(cache_dir / f"{example_session}_sftf_trials.pickle")
    t_df = trials_data["trials"].copy()
    t_df["response"] = trials_data["dff"][:, example_roi]
    angle_range = np.sort(t_df.Angle.unique())
    sf_range = np.sort(t_df.SpatialFrequency.unique())
    tf_range = np.sort(t_df.TemporalFrequency.unique())
    all_angles = np.zeros((len(angle_range), len(sf_range), len(tf_range)))
    for ia, ang in enumerate(angle_range):
        sub_a = t_df[t_df.Angle == ang]
        for isf, sf in enumerate(sf_range):
            for itf, tf in enumerate(tf_range):
                all_angles[ia, isf, itf] = sub_a[
                    (sub_a.SpatialFrequency == sf) & (sub_a.TemporalFrequency == tf)
                ]["response"].mean()
    vmax_sftf = float(np.round(np.nanmax(all_angles), 2))

    # 3. Population correlations & held-out speed curves (cached in cache_dir)
    summary_cache = (
        cache_dir / f"fig3_sftf_summary_v{SPEED_CURVES_VERSION}_n{n_boots}.pickle"
    )
    if summary_cache.exists() and not rebuild:
        summary = pd.read_pickle(summary_cache)
    else:
        stat_speed_depth = _bootstrap_spearman(
            tuned.pref_speed,
            tuned.preferred_depth * 100,
            tuned.session,
            n_boots=n_boots,
        )
        stat_sf_depth = _bootstrap_spearman(
            np.exp(tuned.fit_sf0),
            tuned.preferred_depth * 100,
            tuned.session,
            n_boots=n_boots,
        )

        # Running-speed held-out curves for TF and SF
        rs_models = ["base", "rsg", "rsg_tf", "rsg_sf"]
        pickle_files = [
            f
            for f in sorted(
                (sftf_root / "running_models").glob("*_sftf_running_models.pickle")
            )
            if not f.name.startswith(".")
        ]
        rs_fits = pd.concat(
            [pd.read_pickle(f) for f in pickle_files], ignore_index=True
        )
        fit_cols = [c for c in cells.columns if c.startswith("fit_")]
        rs_fits = rs_fits.merge(
            cells[["session", "roi", "depth_tuned"] + fit_cols],
            on=["session", "roi"],
            how="left",
            validate="one_to_one",
        )
        common_utils.add_rsq_significance(
            rs_fits, {m: f"{m}_test_rsq" for m in rs_models}, verbose=False
        )
        pop_cells = rs_fits[rs_fits.base_test_rsq_sig]

        heldout_rows, p_mod = [], []
        speed_labels = None
        for var in ("tf", "sf"):
            for session, sc in pop_cells.groupby("session"):
                d = np.load(frames_dir / f"{session}.npz")
                cols = [int(np.flatnonzero(d["rois"] == r)[0]) for r in sc.roi]
                sf_v = np.exp(np.unique(d["X"][0].round(4)))
                tf_v = np.exp(np.unique(d["X"][1].round(4)))
                ang_v = np.unique(d["X"][2].round(4))
                kept = sftf_model_kept_pairs(
                    sc,
                    sf_v,
                    tf_v,
                    ang_v,
                    variable=var,
                    threshold=TF_SELECT_THRESHOLD,
                )
                args = (
                    d["X"],
                    d["dff"][:, cols],
                    float(d["log_rs_median"]),
                    SPEED_EDGES_CM_S,
                )
                kw = dict(threshold=TF_SELECT_THRESHOLD, variable=var, kept_pairs=kept)
                # still frames: one more bin ("Static"), aligned and normalised by the running half
                ds = np.load(still_dir / f"{session}.npz")
                assert np.array_equal(ds["rois"], d["rois"]), f"{session}: ROIs differ"
                still_args = (ds["X"], ds["dff"][:, cols], np.nan, None)
                recordings = set(d["recording"]) | set(ds["recording"])
                half = _frame_halves(d["recording"], recordings)
                half_still = _frame_halves(ds["recording"], recordings)
                for h in (0, 1):
                    a, b = half == h, half != h
                    curves_a = population_tf_curves(*args, **kw, select=a, use=a)[0]
                    curves_b, _, speed_labels = population_tf_curves(
                        *args, **kw, select=a, use=b
                    )
                    curves_still = population_tf_curves(
                        *still_args, **kw, use=half_still != h
                    )[0]
                    assert curves_still.shape[2] == curves_b.shape[2]
                    # peak and its value from the running half `a`, applied to the held-out half
                    peak, resp = _align_to_peak(
                        curves_a, np.concatenate([curves_b, curves_still], axis=1)
                    )
                    labels = np.asarray(list(speed_labels) + ["Static"])
                    bin_ids = np.r_[np.arange(len(speed_labels)), -1]
                    idx = np.indices(resp.shape).reshape(3, -1)
                    heldout_rows.append(
                        pd.DataFrame(
                            dict(
                                session=sc.session.to_numpy()[idx[0]],
                                roi=sc.roi.to_numpy()[idx[0]],
                                depth_tuned=sc.depth_tuned.to_numpy()[idx[0]],
                                speed_bin=bin_ids[idx[1]],
                                speed=labels[idx[1]],
                                delta=idx[2] - peak[idx[0]],
                                resp=resp.ravel(),
                                variable=var,
                                fold=h,
                            )
                        )
                    )
                t_sess = pd.read_pickle(cache_dir / f"{session}_sftf_trials.pickle")
                p_mod.append(
                    pd.DataFrame(
                        dict(
                            variable=var,
                            session=session,
                            roi=sc.roi.to_numpy(),
                            depth_tuned=sc.depth_tuned.to_numpy(),
                            p=_freq_modulation_p(
                                t_sess["trials"],
                                t_sess["dff"][:, sc.roi.to_numpy()],
                                var,
                            ),
                        )
                    )
                )
        p_mod = pd.concat(p_mod, ignore_index=True)
        p_mod["freq_tuned"] = p_mod.p < P_FREQ_TUNED

        heldout_pop = (
            pd.concat(heldout_rows, ignore_index=True)
            .merge(
                p_mod[p_mod.freq_tuned][["variable", "session", "roi"]],
                on=["variable", "session", "roi"],
            )
            .groupby(
                [
                    "variable",
                    "depth_tuned",
                    "session",
                    "roi",
                    "speed_bin",
                    "speed",
                    "delta",
                ]
            )
            .resp.mean()
            .reset_index()
        )

        def _boot_ci(values):
            lo, hi = common_utils.get_bootstrap_ci(values.dropna().to_numpy())
            return pd.Series(dict(ci_low=lo[0], ci_high=hi[0]))

        heldout_means = (
            heldout_pop.groupby(
                ["variable", "depth_tuned", "speed_bin", "speed", "delta"]
            )
            .resp.agg(["mean", "sem", "count"])
            .reset_index()
        )
        heldout_means = heldout_means[heldout_means["count"] >= MIN_CELLS_PER_POINT]
        heldout_means = heldout_means.merge(
            heldout_pop.groupby(
                ["variable", "depth_tuned", "speed_bin", "speed", "delta"]
            )
            .resp.apply(_boot_ci)
            .unstack()
            .reset_index(),
            on=["variable", "depth_tuned", "speed_bin", "speed", "delta"],
        )
        heldout_counts = {}
        for (var, is_dt), grp in heldout_pop.groupby(["variable", "depth_tuned"]):
            u = grp[["session", "roi"]].drop_duplicates()
            heldout_counts[(var, is_dt)] = (
                len(u),
                u.session.nunique(),
                u.session.str.split("_").str[0].nunique(),
            )
        mean_peaks = {}
        for (var, is_dt), grp in heldout_pop[heldout_pop.resp.notna()].groupby(
            ["variable", "depth_tuned"]
        ):
            cp = grp.loc[grp.groupby(["session", "roi", "speed_bin"]).resp.idxmax()]
            mean_peaks[(var, is_dt)] = cp.groupby("speed_bin").delta.mean().to_dict()

        summary = dict(
            stat_speed_depth=stat_speed_depth,
            stat_sf_depth=stat_sf_depth,
            heldout_means=heldout_means,
            heldout_counts=heldout_counts,
            mean_peaks=mean_peaks,
            speed_labels=speed_labels,
        )
        pd.to_pickle(summary, summary_cache)

    return dict(
        cells=cells,
        tuned=tuned,
        tuned_other=tuned_other,
        dir_tuned=dir_tuned,
        dir_tuned_other=dir_tuned_other,
        dsi_thr=dsi_thr,
        example_session=example_session,
        example_roi=example_roi,
        ex_depth=ex_depth,
        all_angles=all_angles,
        angle_range=angle_range,
        sf_range=sf_range,
        tf_range=tf_range,
        vmax_sftf=vmax_sftf,
        **summary,
    )


def plot_sftf_3x3_grid(
    fig,
    box,
    all_angles,
    angle_range,
    sf_range,
    tf_range,
    vmax=None,
    fontsize_dict=None,
    cbar_width=0.006,
    cbar_gap=0.005,
):
    """Plot the 8-direction SFxTF heatmaps around a central polar plot inside `box`."""
    fontsize_dict = fontsize_dict or {"title": 7, "label": 7, "tick": 5, "legend": 5}
    x_b, y_b, w_b, h_b = box
    if vmax is None:
        vmax = float(np.round(np.nanmax(all_angles), 2))

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
    sub_w = w_b / 3.45
    sub_h = h_b / 3.45
    gap_x = (w_b - 3 * sub_w) / 2
    gap_y = (h_b - 3 * sub_h) / 2

    polar_ax = fig.add_axes(
        [
            x_b + sub_w + gap_x + sub_w * 0.16,
            y_b + sub_h + gap_y + sub_h * 0.16,
            sub_w * 0.68,
            sub_h * 0.68,
        ],
        projection="polar",
    )
    dir_tuning = np.nanmax(all_angles, axis=(1, 2))
    dir_tuning = np.append(dir_tuning, dir_tuning[0])
    dirs = np.deg2rad(np.append(angle_range, angle_range[0]))

    polar_ax.plot(dirs, dir_tuning, color="k", linewidth=0.8)
    polar_ax.fill_between(dirs, dir_tuning, color="#4A90E2", alpha=0.7)
    polar_ax.set_ylim(0, vmax)
    polar_ax.set_yticks([vmax])
    polar_ax.set_yticklabels([f"{vmax:g}"], fontsize=fontsize_dict["tick"] - 1.5)
    polar_ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    polar_ax.set_xticklabels(
        [f"{int(d)}°" for d in [0, 45, 90, 135, 180, 225, 270, 315]],
        fontsize=fontsize_dict["tick"] - 2,
    )
    polar_ax.tick_params(axis="x", pad=-4)

    last_im = None
    for row in range(3):
        for col in range(3):
            if (row, col) == (1, 1):
                continue
            angle = grid_map[(row, col)]
            angle_idx = np.where(angle_range == angle)[0][0]
            ax_x = x_b + col * (sub_w + gap_x)
            ax_y = y_b + (2 - row) * (sub_h + gap_y)
            ax_dir = fig.add_axes([ax_x, ax_y, sub_w, sub_h])

            last_im = ax_dir.imshow(
                all_angles[angle_idx].T,
                origin="lower",
                vmax=vmax,
                vmin=0,
                cmap="magma",
                aspect="auto",
            )
            ax_dir.tick_params(length=2, pad=1)
            if col == 0:
                ax_dir.set_yticks(range(len(tf_range)))
                ax_dir.set_yticklabels(
                    [f"{t:g}" for t in tf_range], fontsize=fontsize_dict["tick"] - 1
                )
                if row == 1:
                    ax_dir.set_ylabel(
                        "TF (Hz)", fontsize=fontsize_dict["label"], labelpad=1
                    )
            else:
                ax_dir.set_yticks([])
                ax_dir.set_yticklabels([])

            if row == 2:
                ax_dir.set_xticks(range(len(sf_range)))
                ax_dir.set_xticklabels(
                    [f"{s:g}" for s in sf_range],
                    fontsize=fontsize_dict["tick"] - 1,
                    rotation=90,
                )
                if col == 1:
                    ax_dir.set_xlabel(
                        "SF (cpd)", fontsize=fontsize_dict["label"], labelpad=1
                    )
            else:
                ax_dir.set_xticks([])
                ax_dir.set_xticklabels([])

    cax = fig.add_axes([x_b + w_b + cbar_gap, y_b, cbar_width, h_b * 0.5])
    cbar = fig.colorbar(last_im, cax=cax)
    cbar.set_ticks([0, vmax])
    cbar.set_ticklabels(["0", f"{vmax:g}"])
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"], pad=1)
    cbar.ax.set_ylabel(r"$\Delta$F/F", fontsize=fontsize_dict["label"], labelpad=-5)
    return polar_ax, cax


def plot_polar_direction_hist(
    ax,
    values_deg,
    period=360,
    n_bins=8,
    step=False,
    color="k",
    proportion=True,
    lw=1.5,
    fontsize_dict=None,
):
    """Plot polar histogram of preferred directions."""
    fontsize_dict = fontsize_dict or {"title": 7, "label": 7, "tick": 5, "legend": 5}
    width = period / n_bins
    counts, _ = np.histogram(
        (values_deg + width / 2) % period, bins=n_bins, range=(0, period)
    )
    heights = counts / counts.sum() if proportion else counts
    if step:
        theta = np.linspace(0, period, 721)
        r = heights[((theta + width / 2) // width).astype(int) % n_bins]
        ax.plot(np.deg2rad(theta), r, color=color, lw=lw, zorder=3)
    else:
        centres = np.deg2rad(np.arange(n_bins) * width)
        ax.bar(centres, heights, width=np.deg2rad(width) * 0.95, color=color, lw=0)
    ax.set_thetamin(0)
    ax.set_thetamax(period)
    ax.set_xticks(np.deg2rad(np.arange(0, period, 45)))
    ax.set_rlabel_position(period / n_bins / 2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    ax.tick_params(labelsize=fontsize_dict["tick"], pad=1)
    return heights


def _curve_centre_of_mass(delta, response):
    """Centre of mass in octaves (delta) of (response - min of response)."""
    w = np.asarray(response, float) - np.nanmin(response)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    denom = np.sum(w)
    return float(np.sum(w * delta) / denom) if denom > 0 else np.nan


def plot_heldout_speed_curves_panel(
    ax,
    var,
    heldout_means,
    mean_peaks=None,
    speed_labels=(),
    colors=None,
    fontsize_dict=None,
    peak_y=None,
    plot_static=True,
    norm_to_own_peak=False,
    use_com=True,
    is_depth_tuned=True,
):
    """Plot held-out TF or SF tuning curves across running-speed bins for depth-selective or non-depth cells.

    The still frames are the bin -1 of `heldout_means` (`load_fig3_sftf_data`), plotted in black if
    `plot_static`.
    """
    fontsize_dict = fontsize_dict or {"title": 7, "label": 7, "tick": 5, "legend": 5}
    if colors is None:
        colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(speed_labels)))

    m = heldout_means[
        (heldout_means.variable == var) & (heldout_means.depth_tuned == is_depth_tuned)
    ]
    peaks = mean_peaks.get((var, is_depth_tuned), {}) if mean_peaks is not None else {}

    c_stat = m[m.speed_bin == -1].sort_values("delta")
    if plot_static and len(c_stat):
        if norm_to_own_peak:
            stat_min = float(c_stat["mean"].min())
            stat_scale = float(c_stat["mean"].max()) - stat_min
            stat_scale = stat_scale if stat_scale > 0 else 1.0
        else:
            stat_min = 0.0
            stat_scale = 1.0
        ax.errorbar(
            c_stat.delta,
            (c_stat["mean"] - stat_min) / stat_scale,
            yerr=(
                (c_stat["mean"] - c_stat.ci_low) / stat_scale,
                (c_stat.ci_high - c_stat["mean"]) / stat_scale,
            ),
            fmt=".-",
            color="k",
            markeredgecolor="w",
            markeredgewidth=0.3,
            linewidth=1.1,
            markersize=6.0,
            label="Static",
            zorder=2,
        )
        if peak_y is not None:
            stat_pos = (
                _curve_centre_of_mass(c_stat.delta, c_stat["mean"])
                if use_com
                else peaks.get(-1)
            )
            if stat_pos is not None and np.isfinite(stat_pos):
                ax.plot(
                    stat_pos,
                    peak_y,
                    marker="v",
                    ms=3,
                    color="k",
                    lw=0,
                    clip_on=False,
                )

    for b, (label, color) in enumerate(zip(speed_labels, colors)):
        c = m[m.speed_bin == b].sort_values("delta")
        if norm_to_own_peak:
            bin_min = float(c["mean"].min())
            bin_scale = float(c["mean"].max()) - bin_min
            bin_scale = bin_scale if bin_scale > 0 else 1.0
        else:
            bin_min = 0.0
            bin_scale = 1.0
        ax.errorbar(
            c.delta,
            (c["mean"] - bin_min) / bin_scale,
            yerr=(
                (c["mean"] - c.ci_low) / bin_scale,
                (c.ci_high - c["mean"]) / bin_scale,
            ),
            fmt=".-",
            color=color,
            markeredgecolor="w",
            markeredgewidth=0.3,
            linewidth=1,
            markersize=6.0,
            label=label,
        )
        if peak_y is not None:
            bin_pos = (
                _curve_centre_of_mass(c.delta, c["mean"]) if use_com else peaks.get(b)
            )
            if bin_pos is not None and np.isfinite(bin_pos):
                ax.plot(
                    bin_pos,
                    peak_y,
                    marker="v",
                    ms=3,
                    color=color,
                    lw=0,
                    clip_on=False,
                )

    var_label = "TF" if var == "tf" else "SF"
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel(
        f"{var_label} relative to peak {var_label} (octaves)",
        fontsize=fontsize_dict["label"],
    )
    ax.tick_params(labelsize=fontsize_dict["tick"], pad=1)
    ax.spines[["top", "right"]].set_visible(False)
