"""Example neurons for the reviewer grating figure: depth, optic flow and SF/TF tuning.

Port of the example-neuron figure of the thesis (`yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb`,
cell 4) to the current data: `ast_neuropil=False` traces and the cross-validated
`neurons_df_sftf.pickle`. One row per neuron:

- depth tuning (trial means and Gaussian fit) and optic flow tuning per depth, from the
  closed-loop depth protocol;
- mean response to each SF x TF for the 8 directions, around a polar plot of the maximum
  per direction (`grating_plots.plot_sftf_tuning`);
- the same for the fitted model, on the same colour scale (`grating_plots.plot_sftf_fit`).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cottage_analysis.analysis import gratings, spheres
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import depth_selectivity_plots, grating_plots, rsof_plots

PROJECT = "hey2_3d-vision_foodres_20220101"
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}


def load_example_session(flexilims_session, session, photodiode_protocol=5):
    """Trial tables and fits of one grating session.

    Returns:
        dict: `trials_depth` (closed-loop depth protocol, from `spheres.sync_all_recordings`),
            `trials_sftf` (grating trials with one integer column per ROI, from
            `gratings.analyze_grating_responses`), `neurons_df` and `neurons_df_sftf`.
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
    trials_sftf, _ = gratings.analyze_grating_responses(
        project=PROJECT,
        session=session,
        filter_datasets=FILTER_TRACES,
        photodiode_protocol=photodiode_protocol,
        protocol_base="SFTF",
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
        neurons_df_sftf=pd.read_pickle(folder / "neurons_df_sftf.pickle"),
    )


def plot_example_neuron(
    fig,
    data,
    roi,
    top,
    fontsize_dict,
    ylim_precision_base=1,
    ylim_precision=1,
):
    """Plot one example neuron in a band of `fig` whose top is at `top` (figure fraction).

    The band is ~0.65 high: tuning curves on top, raw (left) and fitted (right) SF/TF
    responses below. Layout and settings are those of the thesis figure.
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
    ax = fig.add_axes([0.3, top, 0.15, 0.1])
    rsof_plots.plot_speed_tuning(
        trials_df=data["trials_depth"],
        roi=roi,
        is_closed_loop=1,
        nbins=10,
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
        ax=ax,
    )

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
    vmax = grating_plots.plot_sftf_tuning(
        trials_df=trials_sftf,
        roi=roi,
        plot_x=0.05,
        fontsize_dict=fontsize_dict,
        colorbar=False,
        vmax=None,
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


def speed_binned_responses(X, response, log_rs_median, n_bins=3, edges_cm_s=None):
    """Frame responses of one ROI split into running-speed bins of equal frame counts.

    Args:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - `log_rs_median`, as returned
            by `precompute_data.sftf_running_models.session_frames` (running frames only).
        response (n_frames,): dF/F of the ROI on the same frames.
        log_rs_median (float): median log RS (m/s) subtracted from X[3].
        n_bins (int): number of speed bins (quantiles of log RS), if `edges_cm_s` is None.
        edges_cm_s (array, optional): bin edges in cm/s; frames outside are dropped.

    Returns:
        list of (label, DataFrame): one table per bin with `SpatialFrequency`,
            `TemporalFrequency`, `Angle` (deg) and the response in column 0, the format of
            `grating_plots.plot_sftf_tuning` (which averages the rows of each condition).
    """
    log_rs = X[3] + log_rs_median
    if edges_cm_s is None:
        edges = np.quantile(log_rs, np.linspace(0, 1, n_bins + 1))
        bins = np.clip(np.searchsorted(edges, log_rs, side="right") - 1, 0, n_bins - 1)
    else:
        edges = np.log(np.asarray(edges_cm_s) / 100)
        n_bins = len(edges) - 1
        bins = np.searchsorted(edges, log_rs, side="right") - 1  # -1 or n_bins: outside
    frames = pd.DataFrame(
        {
            "SpatialFrequency": np.exp(X[0]).round(4),
            "TemporalFrequency": np.exp(X[1]).round(4),
            "Angle": np.rad2deg(X[2]).round().astype(int),
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


def tf_curves_by_speed(binned, threshold=0.5):
    """TF tuning per speed bin, over the SF / direction pairs that drive the cell.

    The pairs kept are those whose mean response (over TF and all frames of all bins) is at least
    `threshold` x that of the best pair, so the selection is the same for every bin. For each bin,
    the curve is the mean over kept pairs of the condition mean at each TF, with the SEM across pairs.

    Args:
        binned: output of `speed_binned_responses`.
        threshold (float): fraction of the best pair's response.

    Returns:
        (DataFrame, list): one row per bin and TF (`bin`, `TemporalFrequency`, `mean`, `sem`,
            `n_pairs`), and the kept (SpatialFrequency, Angle) pairs.
    """
    keys = ["SpatialFrequency", "Angle"]
    pair_resp = pd.concat([df for _, df in binned]).groupby(keys)[0].mean()
    kept = list(pair_resp[pair_resp >= threshold * pair_resp.max()].index)
    rows = []
    for label, df in binned:
        sel = df[pd.MultiIndex.from_frame(df[keys]).isin(kept)]
        cond = sel.groupby(keys + ["TemporalFrequency"])[0].mean()
        per_tf = cond.groupby("TemporalFrequency").agg(["mean", "sem", "count"])
        rows.append(
            per_tf.rename(columns={"count": "n_pairs"}).assign(bin=label).reset_index()
        )
    return pd.concat(rows, ignore_index=True), kept


def plot_tf_curves(ax, curves, cmap="viridis", fontsize_dict=None):
    """One TF curve (mean +- SEM across SF / direction pairs) per speed bin, slow to fast."""
    fontsize_dict = fontsize_dict or {"label": 6, "tick": 5, "legend": 5}
    labels = list(dict.fromkeys(curves["bin"]))
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(labels)))
    for label, color in zip(labels, colors):
        c = curves[curves["bin"] == label]
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
    ax.set_xscale("log")
    tf = np.sort(curves.TemporalFrequency.unique())
    ax.set_xticks(tf, [f"{t:g}" for t in tf])
    ax.minorticks_off()
    ax.set_xlabel("Temporal frequency (Hz)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("ΔF/F", fontsize=fontsize_dict["label"])
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
