"""Example neurons for running speed tuning in V1, voluntary running vs motorized treadmill.

Each example shows the speed tuning curve with its 1D Gaussian running speed (RS) fit, next to
the stimulus-period RS/optic flow (OF) response matrix drawn as in the manuscript figures.

1. Closed-Loop Voluntary Running (SpheresPermTubeReward):
   - Panel A: 9 neurons spanning monotonic decreasing through intermediate preferred speeds
     to monotonic increasing (all depths together + ITI blank).

2. Motorized Treadmill (SpheresTubeMotor, Trial-Average Plateau Fit), one session:
   - Panel B: 9 RS-tuned neurons ordered by preferred speed.
   - Panel C: 3 depth-tuned neurons with diagonal RS/OF tuning (RS/OF ratio model R^2 shown).
   - Panel D: 3 OF-tuned neurons (OF model R^2 shown).
Matrices use the motor speed / optic flow levels (`compute_treadmill_rsof_bins`).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

import flexiznam as flz
from cottage_analysis.analysis import (
    common_utils,
    spheres,
    treadmill,
    fit_gaussian_blob,
)
from cottage_analysis.plotting import rsof_plots
from cottage_analysis.plotting.style import CM, FONTSIZE_DICT
from v1_depth_map.paths import get_figures_roots
from v1_depth_map.figure_utils.treadmill import (
    compute_treadmill_rsof_bins,
    load_treadmill_population_neurons_df,
)
from v1_depth_map.figure_utils import trial_averages

# Set plot style
sns.set_theme(style="ticks")
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


def get_output_dir(flexilims_session=None):
    """Return save root on BlackPasspo if available, else local directory."""
    if flexilims_session is not None:
        try:
            _, save_root = get_figures_roots(
                flexilims_session, fig_subdir="running_tuning"
            )
            save_root.mkdir(parents=True, exist_ok=True)
            return save_root
        except Exception:
            pass
    bp_dir = Path(
        "/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/running_tuning"
    )
    if bp_dir.parent.exists():
        bp_dir.mkdir(parents=True, exist_ok=True)
        return bp_dir
    return Path(__file__).resolve().parent


OUTPUT_DIR = get_output_dir()
PROJECT_CL = "hey2_3d-vision_foodres_20220101"
PROJECT_TM = "colasa_3d-vision_revisions"
EXAMPLE_SESSION_CL = "PZAH8.2h_S20230116"
EXAMPLE_SESSION_TM = "PZAG17.3a_S20250402"
# Treadmill RS-tuned examples (panel B), 3 per row, ordered by preferred RS
TM_RS_EXAMPLE_ROIS = [266, 781, 249, 288, 18, 400, 763, 60, 761]


def load_closedloop_population_data(flexilims_session=None):
    """Load closed-loop RS/OF population parquet data (cells only) with preferred RS in cm/s."""
    if flexilims_session is not None:
        try:
            read_root, _ = get_figures_roots(flexilims_session)
            parquet_path = read_root / "fig2" / "results_all_rsof_closedloop.parquet"
        except Exception:
            parquet_path = Path(
                "/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/fig2/results_all_rsof_closedloop.parquet"
            )
    else:
        parquet_path = Path(
            "/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/fig2/results_all_rsof_closedloop.parquet"
        )

    print(f"Loading closed-loop population data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df_cells = df[df["iscell"] == 1].copy()

    # Extract preferred running speed in cm/s for 1D RS Gaussian fit (grs)
    df_cells["popt_grs_x0"] = df_cells["rsof_popt_closedloop_grs"].apply(
        lambda p: p[1] if (p is not None and len(p) >= 4) else np.nan
    )
    df_cells["preferred_RS_grs_cm_s"] = np.exp(df_cells["popt_grs_x0"]) * 100

    return df_cells


def load_treadmill_population_data(flm_tm=None):
    """Load motorized treadmill population data with preferred RS (plateau fit) in cm/s."""
    if flm_tm is None:
        flm_tm = flz.get_flexilims_session(PROJECT_TM)

    print("Loading treadmill population data...")
    pop_df, _, _, _, _ = load_treadmill_population_neurons_df(
        flm_tm, load_simulated=False
    )

    # Preferred RS is stored in m/s
    TA = "_treadmill_trial_average_plateau"
    pop_df["preferred_RS_grs_cm_s"] = pop_df[f"preferred_RS_closedloop_grs{TA}"] * 100

    return pop_df


def load_cl_example_trials_df(session_name=EXAMPLE_SESSION_CL, flexilims_session=None):
    """Load synchronized trials dataframe for the closed-loop example session."""
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(PROJECT_CL)

    print(f"Loading and syncing trials_df for closed-loop session {session_name}...")
    _, trials_df = spheres.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flexilims_session,
        project=PROJECT_CL,
        filter_datasets={"anatomical_only": 3, "ast_neuropil": False},
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=5,
        return_volumes=True,
    )
    return trials_df


def load_tm_example_data(session_name=EXAMPLE_SESSION_TM, flm_tm=None):
    """Load synchronized treadmill trials and trial-average plateau responses."""
    if flm_tm is None:
        flm_tm = flz.get_flexilims_session(PROJECT_TM)

    print(f"Loading and syncing treadmill trials for {session_name}...")
    _, trials_df_tm = treadmill.sync_all_recordings(
        session_name=session_name,
        flexilims_session=flm_tm,
        project=PROJECT_TM,
        filter_datasets={"anatomical_only": 3, "annotated": True},
        recording_type="two_photon",
        photodiode_protocol=5,
        method="plateau",
    )
    red = trial_averages.reduce_trials_to_average(trials_df_tm)
    return trials_df_tm, red


def _fmt_speed(pref):
    return "≤0.5" if pref <= 0.6 else ("≥500" if pref >= 490 else f"{pref:.1f}")


MATRIX_FONTSIZE = dict(
    FONTSIZE_DICT, label=FONTSIZE_DICT["label"] - 1.5, tick=FONTSIZE_DICT["tick"] - 1
)


def _plot_rsof_matrix(fig, ax, trials_df, roi, xlabel, ylabel, **range_kwargs):
    """Stimulus-period RS/OF matrix, drawn as in the manuscript figures (0 to cell max)."""
    ax.set_anchor("W")
    _, _, mat = rsof_plots.plot_RS_OF_matrix(
        trials_df=trials_df,
        roi=roi,
        is_closed_loop=1,
        vmin=0,
        vmax=1,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_width=None,
        fontsize_dict=MATRIX_FONTSIZE,
        ax=ax,
        return_matrix=True,
        **range_kwargs,
    )
    vmax = max(np.ceil(np.nanmax(mat) * 100) / 100, 0.01)
    ax.images[0].set_clim(0, vmax)
    ax.tick_params(axis="both", length=1.5, pad=1)
    ax.apply_aspect()
    rsof_plots.add_rsof_colorbar(fig, ax, ax.images[0], None, 0, vmax, MATRIX_FONTSIZE)


def _example_grid(fig, gs_outer, n_examples):
    """Yield (row, col, tuning_ax, matrix_ax) for a 3-column grid of example cells."""
    for i in range(n_examples):
        row_idx, col_idx = divmod(i, 3)
        gs_cell = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=gs_outer[row_idx, col_idx],
            width_ratios=[1.5, 1.0],
            wspace=0.35,
        )
        yield (
            row_idx,
            col_idx,
            fig.add_subplot(gs_cell[0, 0]),
            fig.add_subplot(gs_cell[0, 1]),
        )


def _cell_title(fig, gs_outer, row_idx, col_idx, text, y=None):
    """Title centred over a tuning-curve + matrix pair."""
    cell_box = gs_outer[row_idx, col_idx].get_position(fig)
    y_pos = cell_box.y1 + 0.004 if y is None else y
    fig.text(
        (cell_box.x0 + cell_box.x1) / 2,
        y_pos,
        text,
        fontsize=FONTSIZE_DICT["legend"] - 1.0,
        ha="center",
        va="bottom",
    )


def _panel_header(fig, y, letter, title):
    """Panel letter and bold title on one line at figure height `y`."""
    fig.text(
        0.02,
        y,
        letter,
        fontsize=FONTSIZE_DICT["panel"],
        fontweight="bold",
        ha="left",
        va="center",
    )
    fig.text(
        0.05,
        y,
        title,
        fontsize=FONTSIZE_DICT["title"],
        fontweight="bold",
        ha="left",
        va="center",
    )


def generate_figure(
    df_cells_cl, trials_df_cl, df_cells_tm, trials_df_tm, red_tm, out_dir=OUTPUT_DIR
):
    """Generate and save the example-cell figure (A: closed loop, B-D: treadmill)."""
    fig_width = 18.5 * CM
    fig_height_cm = 34.3
    fig = plt.figure(figsize=(fig_width, fig_height_cm * CM))

    def fy(cm_from_top):
        """Figure y fraction from a distance (cm) below the top edge."""
        return 1 - cm_from_top / fig_height_cm

    color_rs = "#D95F02"  # vermilion
    color_tm = "#1B9E77"  # teal / green
    # Example-cell rows (cm): axes height, gap between rows (cell titles + ticks), space
    # above the first row for the panel header and cell titles, and below the last row
    # for the x labels before the next panel header.
    row_h, row_gap, head_gap, foot_gap = 2.6, 1.15, 0.95, 1.25

    def example_gs(header_y, n_rows):
        """3-column GridSpec of `n_rows` example rows below a panel header at `header_y` cm."""
        top = header_y + head_gap
        bottom = top + n_rows * row_h + (n_rows - 1) * row_gap
        gs = gridspec.GridSpec(
            n_rows,
            3,
            left=0.08,
            right=0.95,
            top=fy(top),
            bottom=fy(bottom),
            hspace=row_gap / row_h,
            wspace=0.45,
        )
        return gs, bottom + foot_gap

    # -------------------------------------------------------------------------
    # Panel A: 9 Example Single Neurons (Closed Loop) with their RS/OF matrices
    # -------------------------------------------------------------------------
    header_y = 0.45
    _panel_header(
        fig, fy(header_y), "A", f"Voluntary running, closed loop ({EXAMPLE_SESSION_CL})"
    )

    legend_cl = [
        Line2D(
            [0],
            [0],
            color=color_rs,
            ls="",
            marker="o",
            markersize=3.0,
            label="Stimulus (all depths)",
        ),
        Line2D([0], [0], color=color_rs, lw=1.6, ls="-", label="1D RS fit"),
        Line2D(
            [0], [0], color="#7F7F7F", lw=1.0, ls="--", label="Blank baseline (ITI)"
        ),
    ]
    fig.legend(
        handles=legend_cl,
        loc="upper right",
        bbox_to_anchor=(0.96, fy(header_y - 0.25)),
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE_DICT["legend"],
    )

    gs_a, header_y = example_gs(header_y, 3)

    example_rois_cl = [2, 744, 318, 644, 261, 518, 88, 638, 519]
    sess_cells_cl = df_cells_cl[df_cells_cl["session"] == EXAMPLE_SESSION_CL]

    cl_trials = trials_df_cl[trials_df_cl.closed_loop == 1]
    rs_stim_all = np.concatenate(cl_trials["RS_stim"].values) * 100
    rs_blank_all = np.concatenate(cl_trials["RS_blank"].values) * 100
    dff_stim_all = np.vstack(cl_trials["dff_stim"].values)
    dff_blank_all = np.vstack(cl_trials["dff_blank"].values)

    speed_bins_cl = np.linspace(0, 100, 13)
    bin_centers_cl = (speed_bins_cl[:-1] + speed_bins_cl[1:]) / 2
    smooth_rs_cl = np.linspace(1, 100, 100)
    smooth_log_rs_cl = np.log(smooth_rs_cl / 100)

    for (row_idx, col_idx, ax, ax_m), roi in zip(
        _example_grid(fig, gs_a, len(example_rois_cl)), example_rois_cl
    ):
        cell_info = sess_cells_cl[sess_cells_cl["roi"] == roi].iloc[0]
        popt_grs = cell_info["rsof_popt_closedloop_grs"]
        rsq_grs = cell_info["rsof_test_rsq_closedloop_grs"]
        pref_grs = cell_info["preferred_RS_grs_cm_s"]

        m_stim, _, ci_stim = rsof_plots.calculate_speed_tuning(
            rs_stim_all, dff_stim_all[:, roi], bins=speed_bins_cl, smoothing_sd=1
        )
        _, sm_blank, _ = rsof_plots.calculate_speed_tuning(
            rs_blank_all, dff_blank_all[:, roi], bins=speed_bins_cl, smoothing_sd=1
        )

        # Plot stimulus points (no connecting line between points)
        ax.errorbar(
            bin_centers_cl,
            m_stim,
            yerr=np.abs(ci_stim.T - m_stim),
            fmt="o",
            color=color_rs,
            ms=2.3,
            capsize=1.0,
            linewidth=0.6,
            zorder=4,
        )

        # Plot 1D RS Gaussian fit curve
        if popt_grs is not None and len(popt_grs) >= 4:
            fit_grs_cl = fit_gaussian_blob.gaussian_1d(
                smooth_log_rs_cl, *popt_grs, min_sigma=0.25
            )
            ax.plot(smooth_rs_cl, fit_grs_cl, color=color_rs, lw=1.5, ls="-", zorder=3)

        # Blank baseline across ITI
        ax.plot(bin_centers_cl, sm_blank, color="#7F7F7F", lw=1.0, ls="--", zorder=1)

        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"] - 1.5)
        sns.despine(ax=ax, offset=2, trim=True)
        if row_idx == 2:
            ax.set_xlabel("Speed (cm/s)", fontsize=FONTSIZE_DICT["label"] - 1)
        if col_idx == 0:
            ax.set_ylabel(r"$\Delta$F/F", fontsize=FONTSIZE_DICT["label"])

        _plot_rsof_matrix(
            fig,
            ax_m,
            trials_df_cl,
            roi,
            xlabel="RS (cm/s)" if row_idx == 2 else "",
            ylabel="OF (°/s)" if col_idx == 0 else "",
        )
        _cell_title(
            fig,
            gs_a,
            row_idx,
            col_idx,
            f"ROI {roi}\n$r_0$={_fmt_speed(pref_grs)} cm/s, $R^2$={rsq_grs:.2f}",
        )

    # =========================================================================
    # MOTORIZED TREADMILL (TRIAL-AVERAGE PLATEAU FIT)
    # =========================================================================

    TA = "_treadmill_trial_average_plateau"
    sess_cells_tm = df_cells_tm[df_cells_tm["session"] == EXAMPLE_SESSION_TM].set_index(
        "roi"
    )

    rs_tm_trials = red_tm["rs"] * 100  # cm/s
    dff_tm_trials = red_tm["dff"]  # (n_trials, n_neurons)

    # Nominal motor speed bins
    speed_bins_tm = np.array([0, 5, 11, 22, 45, 90])
    bin_centers_tm = np.array([3.8, 7.6, 15.2, 30.5, 60.8])
    rs_idx = np.digitize(rs_tm_trials, speed_bins_tm) - 1
    smooth_rs_grid = np.linspace(2, 70, 100)
    smooth_log_rs = np.log(smooth_rs_grid / 100)

    # Blank responses for treadmill trials
    cl_tm_trials = trials_df_tm[trials_df_tm.closed_loop == 1]

    # Matrix bins at the motor speed / optic flow levels, as in figure 3
    rs_bins_tm, of_bins_tm, tick_dict_tm = compute_treadmill_rsof_bins(trials_df_tm)
    tm_range_kwargs = dict(
        log_range={"log_base": 2},
        rs_bins=rs_bins_tm,
        of_bins=of_bins_tm,
        tick_dict=tick_dict_tm,
    )

    def plot_treadmill_examples(gs, rois, extra_model=None):
        """Treadmill speed tuning + RS/OF matrix for `rois`, 3 per row of `gs`."""
        n_rows = gs.get_geometry()[0]
        for (row_idx, col_idx, ax, ax_m), roi in zip(
            _example_grid(fig, gs, len(rois)), rois
        ):
            cell_row = sess_cells_tm.loc[roi]
            popt_grs = cell_row[f"rsof_popt_closedloop_grs{TA}"]
            rsq_grs = cell_row[f"rsof_test_rsq_closedloop_grs{TA}"]
            pref_grs = cell_row["preferred_RS_grs_cm_s"]

            # Trial average stimulus responses across the 5 motor speed levels
            means_stim, sems_stim = [], []
            for b_idx in range(len(bin_centers_tm)):
                vals_all = dff_tm_trials[rs_idx == b_idx, roi]
                means_stim.append(vals_all.mean())
                sems_stim.append(vals_all.std() / np.sqrt(len(vals_all)))

            ax.errorbar(
                bin_centers_tm,
                np.array(means_stim),
                yerr=np.array(sems_stim),
                fmt="o",
                color=color_tm,
                ms=2.4,
                capsize=1.0,
                linewidth=0.6,
                zorder=4,
            )

            # Treadmill 1D Gaussian fit curve
            if popt_grs is not None and len(popt_grs) >= 4:
                fit_grs_tm = fit_gaussian_blob.gaussian_1d(
                    smooth_log_rs, *popt_grs, min_sigma=0.25
                )
                ax.plot(
                    smooth_rs_grid, fit_grs_tm, color=color_tm, lw=1.3, ls="-", zorder=3
                )

            # Blank baseline across ITI
            blank_vals = [
                np.nanmean(d[:, roi]) for d in cl_tm_trials["dff_blank"] if len(d)
            ]
            if len(blank_vals):
                ax.axhline(
                    np.nanmean(blank_vals), color="#7F7F7F", lw=1.0, ls="--", zorder=1
                )

            ax.set_xlim(-2, 75)
            ax.set_xticks([0, 30, 60])
            ax.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"] - 1.5)
            sns.despine(ax=ax, offset=2, trim=True)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Speed (cm/s)", fontsize=FONTSIZE_DICT["label"] - 1)
            if col_idx == 0:
                ax.set_ylabel(r"$\Delta$F/F", fontsize=FONTSIZE_DICT["label"])

            _plot_rsof_matrix(
                fig,
                ax_m,
                trials_df_tm,
                roi,
                xlabel="RS (cm/s)" if row_idx == n_rows - 1 else "",
                ylabel="OF (°/s)" if col_idx == 0 else "",
                **tm_range_kwargs,
            )
            title = (
                f"ROI {roi}\nRS: $r_0$={_fmt_speed(pref_grs)} cm/s, $R^2$={rsq_grs:.2f}"
            )
            if extra_model is not None:
                model, label = extra_model
                title += f"; {label} $R^2$={cell_row[f'rsof_test_rsq_closedloop_{model}{TA}']:.2f}"
            _cell_title(fig, gs, row_idx, col_idx, title)

    # -------------------------------------------------------------------------
    # Panel B: 9 RS-tuned example neurons
    # -------------------------------------------------------------------------
    _panel_header(
        fig,
        fy(header_y),
        "B",
        f"Motorized treadmill, trial-average plateau ({EXAMPLE_SESSION_TM})",
    )

    legend_tm = [
        Line2D(
            [0],
            [0],
            color=color_tm,
            ls="",
            marker="o",
            markersize=3.0,
            label="Stimulus (all OFs)",
        ),
        Line2D([0], [0], color=color_tm, lw=1.4, ls="-", label="1D RS fit"),
        Line2D([0], [0], color="#7F7F7F", lw=1.0, ls="--", label="Blank baseline"),
    ]
    fig.legend(
        handles=legend_tm,
        loc="upper right",
        bbox_to_anchor=(0.96, fy(header_y - 0.25)),
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE_DICT["legend"],
    )

    gs_b, header_y = example_gs(header_y, 3)
    plot_treadmill_examples(gs_b, TM_RS_EXAMPLE_ROIS)

    # -------------------------------------------------------------------------
    # Panel C: 3 depth-tuned neurons (diagonal RS/OF tuning)
    # -------------------------------------------------------------------------
    _panel_header(fig, fy(header_y), "C", "Depth-tuned neurons (diagonal RS/OF tuning)")
    gs_c, header_y = example_gs(header_y, 1)
    plot_treadmill_examples(gs_c, [169, 49, 370], extra_model=("gratio", "RS/OF ratio"))

    # -------------------------------------------------------------------------
    # Panel D: 3 optic flow tuned neurons
    # -------------------------------------------------------------------------
    _panel_header(fig, fy(header_y), "D", "Optic flow tuned neurons")
    gs_d, _ = example_gs(header_y, 1)
    plot_treadmill_examples(gs_d, [297, 318, 277], extra_model=("gof", "OF"))

    # Save SVG, PDF and PNG
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "running_tuning_figure.png"
    out_pdf = out_dir / "running_tuning_figure.pdf"

    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, transparent=True)
    print(f"Saved figure to:\n  {out_png}\n  {out_pdf}")

    return fig


def generate_rs_scatter_figure(df_cells_tm, trials_df_tm, rois, out_dir=OUTPUT_DIR):
    """Per-frame stimulus dF/F vs log running speed on the treadmill, one axis per cell.

    Frames are filtered as for the RS/OF matrices (finite RS/OF, |RS - motor| ratio < 0.3,
    RS > 0 for the log axis). The treadmill trial-average 1D RS fit is overlaid.
    """
    TA = "_treadmill_trial_average_plateau"
    color_tm = "#1B9E77"
    sess_cells_tm = df_cells_tm[df_cells_tm["session"] == EXAMPLE_SESSION_TM].set_index(
        "roi"
    )

    tm_trials = trials_df_tm[trials_df_tm.closed_loop == 1]
    rs = np.concatenate(tm_trials["RS_stim"].values) * 100  # cm/s
    of = np.concatenate(tm_trials["OF_stim"].values)
    rs2motor = np.concatenate(tm_trials["max_abs_rs2motor_diff_ratio_stim"].values)
    dff = np.vstack(tm_trials["dff_stim"].values)
    valid = np.isfinite(rs) & np.isfinite(of) & (rs > 0) & (rs2motor < 0.3)
    rs, dff = rs[valid], dff[valid]

    smooth_rs = np.logspace(np.log10(rs.min()), np.log10(rs.max()), 200)

    fig, axes = plt.subplots(1, len(rois), figsize=(18.5 * CM, 5.5 * CM))
    for ax, roi in zip(np.atleast_1d(axes), rois):
        cell_row = sess_cells_tm.loc[roi]
        ax.scatter(
            rs,
            dff[:, roi],
            s=1.5,
            color=color_tm,
            alpha=0.05,
            edgecolor="none",
            rasterized=True,
        )
        popt_grs = cell_row[f"rsof_popt_closedloop_grs{TA}"]
        if popt_grs is not None and len(popt_grs) >= 4:
            fit = fit_gaussian_blob.gaussian_1d(
                np.log(smooth_rs / 100), *popt_grs, min_sigma=0.25
            )
            ax.plot(smooth_rs, fit, color="k", lw=1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100])
        ax.set_xticklabels(["1", "10", "100"])
        ax.set_xlabel("Running speed (cm/s)", fontsize=FONTSIZE_DICT["label"])
        ax.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"])
        ax.set_title(
            f"ROI {roi}\n$r_0$={_fmt_speed(cell_row['preferred_RS_grs_cm_s'])} cm/s, "
            f"$R^2$={cell_row[f'rsof_test_rsq_closedloop_grs{TA}']:.2f}",
            fontsize=FONTSIZE_DICT["legend"],
        )
        sns.despine(ax=ax, offset=2)
    np.atleast_1d(axes)[0].set_ylabel(r"$\Delta$F/F", fontsize=FONTSIZE_DICT["label"])
    fig.suptitle(
        f"Treadmill frames, stimulus period ({EXAMPLE_SESSION_TM}, N={len(rs):,} frames)",
        fontsize=FONTSIZE_DICT["title"],
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "running_tuning_rs_scatter.png"
    out_pdf = out_dir / "running_tuning_rs_scatter.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, transparent=True, dpi=300)
    print(f"Saved figure to:\n  {out_png}\n  {out_pdf}")
    return fig


def generate_pref_rs_histogram(df_cells_cl, df_cells_tm, out_dir=OUTPUT_DIR):
    """Proportion of cells per preferred running speed (1D RS fit), one axis per cell group.

    Groups: cells with a significant 1D RS fit, all cells, and depth-selective cells
    (one-sided Spearman on the closed-loop depth tuning of each dataset's spheres protocol).

    Bins: octave-wide bins centred on the treadmill motor speeds, plus one bin below the
    slowest and one above the fastest of them; those two outer bins are drawn offset from
    the motor-speed bins. Closed loop is the original dataset; treadmill is the
    trial-average plateau fit. RS significance: 95th percentile of the empirical null of the
    cross-validated R^2, computed within each dataset.
    """
    TA = "_treadmill_trial_average_plateau"
    motor_speeds = np.array([3.81, 7.62, 15.25, 30.5, 61.0])
    edges = np.concatenate(
        [[0], motor_speeds / np.sqrt(2), [motor_speeds[-1] * np.sqrt(2), np.inf]]
    )
    labels = (
        [f"<{edges[1]:.1f}"] + [f"{v:g}" for v in motor_speeds] + [f">{edges[-2]:.1f}"]
    )

    # Cell groups per dataset: all cells, significant 1D RS fit, depth-selective
    # (closed-loop spheres for the original data, treadmill spheres for the treadmill)
    groups = ["RS-tuned cells", "All cells", "Depth-selective cells"]
    datasets = []
    for df, rsq_col, depth_cols, label, color in [
        (
            df_cells_cl,
            "rsof_test_rsq_closedloop_grs",
            (
                "depth_tuning_test_spearmanr_rval_closedloop",
                "depth_tuning_test_spearmanr_pval_closedloop",
            ),
            "Closed loop",
            "#D95F02",
        ),
        (
            df_cells_tm,
            f"rsof_test_rsq_closedloop_grs{TA}",
            (
                "depth_tuning_test_spearmanr_rval_closedloop_treadmill",
                "depth_tuning_test_spearmanr_pval_closedloop_treadmill",
            ),
            "Treadmill",
            "#1B9E77",
        ),
    ]:
        sel_df = df[[rsq_col, *depth_cols, "preferred_RS_grs_cm_s"]].copy()
        common_utils.add_rsq_significance(sel_df, {"grs": rsq_col}, percentile=95)
        common_utils.add_one_sided_spearman_significance(
            sel_df, rval_col=depth_cols[0], pval_col=depth_cols[1], out_col="is_depth"
        )
        masks = {
            "RS-tuned cells": sel_df[f"{rsq_col}_sig"],
            "All cells": np.ones(len(sel_df), dtype=bool),
            "Depth-selective cells": sel_df["is_depth"],
        }
        for group in groups:
            pref = sel_df.loc[masks[group], "preferred_RS_grs_cm_s"].dropna().values
            counts, _ = np.histogram(pref, bins=edges)
            datasets.append((group, label, color, counts / counts.sum(), len(pref)))
            print(
                f"{label}, {group}: N={len(pref)}, "
                + ", ".join(
                    f"{lab}: {c / counts.sum():.3f}" for lab, c in zip(labels, counts)
                )
            )

    fig, axes = plt.subplots(1, len(groups), figsize=(18 * CM, 6.5 * CM), sharey=True)
    # Motor-speed bins at 0..4; outer bins one bin further out plus a gap
    gap = 0.5
    n_mid = len(motor_speeds)
    x_centers = np.concatenate([[-1 - gap], np.arange(n_mid), [n_mid + gap]])
    for ax, group in zip(axes, groups):
        for _, label, color, props, n in [d for d in datasets if d[0] == group]:
            ax.stairs(
                props[1:-1],
                np.arange(n_mid + 1) - 0.5,
                color=color,
                lw=1.5,
                label=f"{label} (N={n:,})",
            )
            for x, prop in [(x_centers[0], props[0]), (x_centers[-1], props[-1])]:
                ax.stairs([prop], [x - 0.5, x + 0.5], color=color, lw=1.5)
        # despine before setting ticks: moving the spines resets the tick labels
        sns.despine(ax=ax, offset=2)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            labels, fontsize=FONTSIZE_DICT["tick"], rotation=45, ha="right"
        )
        ax.tick_params(axis="y", labelsize=FONTSIZE_DICT["tick"])
        ax.set_xlabel("Preferred running speed (cm/s)", fontsize=FONTSIZE_DICT["label"])
        ax.set_title(group, fontsize=FONTSIZE_DICT["title"], pad=14)
        ax.legend(
            frameon=False,
            fontsize=FONTSIZE_DICT["legend"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=2,
            handlelength=1.2,
            columnspacing=0.8,
        )
    axes[0].set_ylabel("Proportion of cells", fontsize=FONTSIZE_DICT["label"])
    axes[0].set_ylim(0, None)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "running_tuning_pref_rs_hist.png"
    out_pdf = out_dir / "running_tuning_pref_rs_hist.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, transparent=True)
    print(f"Saved figure to:\n  {out_png}\n  {out_pdf}")
    return fig


def generate_depth_pref_summary_figure(
    df_cells_cl,
    trials_df_cl,
    df_cells_tm,
    trials_df_tm,
    red_tm,
    out_dir=OUTPUT_DIR,
    flm_tm=None,
):
    """Generate summary figure: preferred RS histogram (all cells) + CL/TM examples.

    Top row:
      - Left (cols 0-1): Step histogram of preferred running speed for all cells
        (closed loop vs treadmill).
      - Right (cols 2-3): 2 Closed-loop example cells (ROIs 644 and 518) with 1D RS tuning curves
        and RS/OF matrices.
    Bottom row (cols 0-3):
      - 4 Treadmill depth-tuned example cells (ROIs 49, 249, 41, 361) with 1D RS tuning curves
        and RS/OF matrices.
    """
    fig_width = 18.5 * CM
    fig_height_cm = 10.5
    fig = plt.figure(figsize=(fig_width, fig_height_cm * CM))

    def fy(cm_from_top):
        return 1 - cm_from_top / fig_height_cm

    color_cl = "#D95F02"  # vermilion
    color_tm = "#1B9E77"  # teal / green

    # GridSpec with 2 rows and 4 main columns
    gs = gridspec.GridSpec(
        2,
        4,
        left=0.065,
        right=0.94,
        top=fy(1.3),
        bottom=fy(9.8),
        hspace=0.68,
        wspace=0.45,
    )

    y_head1 = fy(0.4)
    y_head2 = fy(5.5)
    TA = "_treadmill_trial_average_plateau"

    def _plot_summary_matrix(
        ax, trials_df, roi, xlabel, ylabel, is_tm=False, **extra_kwargs
    ):
        """Plot RS/OF matrix with compact custom colorbar matching matrix height exactly."""
        ax.set_anchor("W")
        kwargs = dict(
            trials_df=trials_df,
            roi=roi,
            is_closed_loop=1,
            vmin=0,
            vmax=1,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_width=None,
            fontsize_dict=MATRIX_FONTSIZE,
            ax=ax,
            return_matrix=True,
        )
        if is_tm:
            kwargs.update(extra_kwargs)
        _, _, mat = rsof_plots.plot_RS_OF_matrix(**kwargs)
        vmax = max(np.ceil(np.nanmax(mat) * 100) / 100, 0.01)
        ax.images[0].set_clim(0, vmax)
        ax.tick_params(axis="both", length=1.5, pad=1)
        ax.apply_aspect()

        # Custom neat colorbar matching matrix height exactly
        pos = ax.get_position(original=False)
        cax = fig.add_axes([pos.x1 + 0.004, pos.y0, 0.005, pos.height])
        cbar = fig.colorbar(ax.images[0], cax=cax)
        cbar.set_ticks([0, vmax])
        cax.tick_params(labelsize=FONTSIZE_DICT["legend"] - 2.5, length=1.5, pad=1)
        return pos

    # -------------------------------------------------------------------------
    # Panel A: All cells speed modulation (2 separate histograms: Closed loop & Treadmill)
    # -------------------------------------------------------------------------
    _panel_header(fig, y_head1, "A", "All cells")

    # Split gs[0, 0:2] into 2 subplots: ax_cl (col 0) and ax_tm (col 1)
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0:2], wspace=0.48)
    ax_cl = fig.add_subplot(gs_a[0, 0])
    ax_tm = fig.add_subplot(gs_a[0, 1])

    gap = 0.5
    common_ylim = (0, 0.32)

    # 1. Closed loop histogram:
    # Mouse voluntarily runs slower (down to ~0.7 cm/s) and faster (up to >150 cm/s) than the treadmill grid.
    # Bounds:
    # Monotonic decrease: < 2.69 cm/s (< 3.81 / sqrt(2))
    # Inner speeds: 3.81, 7.62, 15.25, 30.5, 61.0 cm/s
    # Monotonic increase: >= 86.27 cm/s (>= 61.0 * sqrt(2))
    motor_speeds_cl = np.array([3.81, 7.62, 15.25, 30.5, 61.0])
    edges_cl = np.concatenate(
        [[0], motor_speeds_cl / np.sqrt(2), [motor_speeds_cl[-1] * np.sqrt(2), np.inf]]
    )
    labels_cl = ["Decr.", "3.8", "7.6", "15.2", "30.5", "61.0", "Incr."]

    n_mid_cl = len(motor_speeds_cl)  # 5
    x_centers_cl = np.concatenate([[-1 - gap], np.arange(n_mid_cl), [n_mid_cl + gap]])

    pref_cl = df_cells_cl["preferred_RS_grs_cm_s"].dropna().values
    counts_cl, _ = np.histogram(pref_cl, bins=edges_cl)
    props_cl = counts_cl / counts_cl.sum()
    n_cl = len(pref_cl)
    pct_nm_cl = props_cl[1:-1].sum() * 100

    ax_cl.bar(
        x_centers_cl,
        props_cl,
        width=1.0,
        color=color_cl,
        edgecolor="white",
        linewidth=1.2,
    )

    sns.despine(ax=ax_cl, offset=2)
    ax_cl.minorticks_off()
    ax_cl.set_xlim(x_centers_cl[0] - 0.9, x_centers_cl[-1] + 0.9)
    ax_cl.set_xticks(x_centers_cl)
    ax_cl.set_xticklabels(
        labels_cl, fontsize=FONTSIZE_DICT["tick"] - 1.0, rotation=35, ha="right"
    )
    ax_cl.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"] - 0.5)
    ax_cl.set_xlabel(r"Speed / $r_0$ (cm/s)", fontsize=FONTSIZE_DICT["label"] - 0.5)
    ax_cl.set_ylabel("Proportion of cells", fontsize=FONTSIZE_DICT["label"] - 0.5)
    ax_cl.set_ylim(common_ylim)
    ax_cl.set_title(
        f"Closed loop (N={n_cl:,})\n{pct_nm_cl:.1f}% non-monotonic",
        fontsize=FONTSIZE_DICT["title"] - 1.0,
        color=color_cl,
        pad=4,
    )

    # 2. Treadmill histogram:
    # Motor speeds are discrete: 3.81, 7.62, 15.25, 30.5, 61.0 cm/s
    # Bounds:
    # Monotonic decrease: < 5.39 cm/s (< 3.81 * sqrt(2), as 3.81 is the slowest tested speed)
    # Inner speeds: 7.62, 15.25, 30.5 cm/s
    # Monotonic increase: >= 43.13 cm/s (>= 61.0 / sqrt(2), as 61.0 is the fastest tested speed)
    motor_speeds_7 = np.array([3.81, 7.62, 15.25, 30.5, 61.0])
    edges_7 = np.concatenate(
        [[0], motor_speeds_7 / np.sqrt(2), [motor_speeds_7[-1] * np.sqrt(2), np.inf]]
    )
    edges_tm = np.array([0, edges_7[2], edges_7[3], edges_7[4], edges_7[5], np.inf])
    labels_tm = ["Decr.", "7.6", "15.2", "30.5", "Incr."]
    inner_speeds_tm = np.array([7.62, 15.25, 30.5])
    n_mid_tm = len(inner_speeds_tm)  # 3
    x_centers_tm = np.concatenate([[-1 - gap], np.arange(n_mid_tm), [n_mid_tm + gap]])

    pref_tm = df_cells_tm["preferred_RS_grs_cm_s"].dropna().values
    counts_tm, _ = np.histogram(pref_tm, bins=edges_tm)
    props_tm = counts_tm / counts_tm.sum()
    n_tm = len(pref_tm)
    pct_nm_tm = props_tm[1:-1].sum() * 100

    ax_tm.bar(
        x_centers_tm,
        props_tm,
        width=1.0,
        color=color_tm,
        edgecolor="white",
        linewidth=1.2,
    )

    sns.despine(ax=ax_tm, offset=2)
    ax_tm.minorticks_off()
    ax_tm.set_xlim(x_centers_tm[0] - 0.9, x_centers_tm[-1] + 0.9)
    ax_tm.set_xticks(x_centers_tm)
    ax_tm.set_xticklabels(
        labels_tm, fontsize=FONTSIZE_DICT["tick"] - 1.0, rotation=35, ha="right"
    )
    ax_tm.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"] - 0.5)
    ax_tm.set_xlabel(r"Speed / $r_0$ (cm/s)", fontsize=FONTSIZE_DICT["label"] - 0.5)
    ax_tm.set_ylabel("Proportion of cells", fontsize=FONTSIZE_DICT["label"] - 0.5)
    ax_tm.set_ylim(common_ylim)
    ax_tm.set_title(
        f"Treadmill (N={n_tm:,})\n{pct_nm_tm:.1f}% non-monotonic",
        fontsize=FONTSIZE_DICT["title"] - 1.0,
        color=color_tm,
        pad=4,
    )

    # -------------------------------------------------------------------------
    # Panel B: 2 Closed-Loop Examples (ROIs 644, 518)
    # -------------------------------------------------------------------------
    box_b = gs[0, 2].get_position(fig)
    fig.text(
        box_b.x0 - 0.025,
        y_head1,
        "B",
        fontsize=FONTSIZE_DICT["panel"],
        fontweight="bold",
        ha="left",
        va="center",
    )
    fig.text(
        box_b.x0,
        y_head1,
        f"Voluntary running ({EXAMPLE_SESSION_CL})",
        fontsize=FONTSIZE_DICT["title"],
        fontweight="bold",
        ha="left",
        va="center",
    )

    legend_cl = [
        Line2D(
            [0],
            [0],
            color=color_cl,
            ls="",
            marker="o",
            markersize=2.6,
            label="Stimulus",
        ),
        Line2D([0], [0], color=color_cl, lw=1.4, ls="-", label="1D RS fit"),
        Line2D([0], [0], color="#7F7F7F", lw=1.0, ls="--", label="Blank (ITI)"),
    ]
    fig.legend(
        handles=legend_cl,
        loc="upper right",
        bbox_to_anchor=(0.95, y_head1 + 0.005),
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE_DICT["legend"] - 1.5,
        handlelength=1.1,
        columnspacing=0.8,
    )

    cl_example_rois = [644, 518]
    sess_cells_cl = df_cells_cl[df_cells_cl["session"] == EXAMPLE_SESSION_CL]

    cl_trials = trials_df_cl[trials_df_cl.closed_loop == 1]
    rs_stim_all = np.concatenate(cl_trials["RS_stim"].values) * 100
    rs_blank_all = np.concatenate(cl_trials["RS_blank"].values) * 100
    dff_stim_all = np.vstack(cl_trials["dff_stim"].values)
    dff_blank_all = np.vstack(cl_trials["dff_blank"].values)

    speed_bins_cl = np.linspace(0, 100, 13)
    bin_centers_cl = (speed_bins_cl[:-1] + speed_bins_cl[1:]) / 2
    smooth_rs_cl = np.linspace(1, 100, 100)
    smooth_log_rs_cl = np.log(smooth_rs_cl / 100)

    for i, roi in enumerate(cl_example_rois):
        col_idx = 2 + i
        gs_cell = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[0, col_idx], width_ratios=[1.3, 1.0], wspace=0.42
        )
        ax = fig.add_subplot(gs_cell[0, 0])
        ax_m = fig.add_subplot(gs_cell[0, 1])

        cell_info = sess_cells_cl[sess_cells_cl["roi"] == roi].iloc[0]
        popt_grs = cell_info["rsof_popt_closedloop_grs"]
        rsq_grs = cell_info["rsof_test_rsq_closedloop_grs"]
        pref_grs = cell_info["preferred_RS_grs_cm_s"]

        m_stim, _, ci_stim = rsof_plots.calculate_speed_tuning(
            rs_stim_all, dff_stim_all[:, roi], bins=speed_bins_cl, smoothing_sd=1
        )
        _, sm_blank, _ = rsof_plots.calculate_speed_tuning(
            rs_blank_all, dff_blank_all[:, roi], bins=speed_bins_cl, smoothing_sd=1
        )

        ax.errorbar(
            bin_centers_cl,
            m_stim,
            yerr=np.abs(ci_stim.T - m_stim),
            fmt="o",
            color=color_cl,
            ms=2.3,
            capsize=1.0,
            linewidth=0.6,
            zorder=4,
        )

        if popt_grs is not None and len(popt_grs) >= 4:
            fit_grs_cl = fit_gaussian_blob.gaussian_1d(
                smooth_log_rs_cl, *popt_grs, min_sigma=0.25
            )
            ax.plot(smooth_rs_cl, fit_grs_cl, color=color_cl, lw=1.4, ls="-", zorder=3)

        ax.plot(bin_centers_cl, sm_blank, color="#7F7F7F", lw=1.0, ls="--", zorder=1)

        ax.set_xlim(-2, 102)
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"] - 1.5)
        sns.despine(ax=ax, offset=2, trim=True)
        ax.set_xlabel("Speed (cm/s)", fontsize=FONTSIZE_DICT["label"] - 1)
        if i == 0:
            ax.set_ylabel(r"$\Delta$F/F", fontsize=FONTSIZE_DICT["label"])

        pos_m = _plot_summary_matrix(
            ax_m,
            trials_df_cl,
            roi,
            xlabel="RS (cm/s)",
            ylabel="OF (°/s)" if i == 0 else "",
        )
        pos_ax = ax.get_position(original=False)
        ax.set_position([pos_ax.x0, pos_m.y0, pos_ax.width, pos_m.height])
        pos_m_cl_last = pos_m
        _cell_title(
            fig,
            gs,
            0,
            col_idx,
            f"ROI {roi}\n$r_0$={_fmt_speed(pref_grs)} cm/s, $R^2$={rsq_grs:.2f}",
            y=pos_m.y1 + 0.005,
        )

    if pos_m_cl_last is not None:
        pos_c = ax_cl.get_position(original=False)
        ax_cl.set_position(
            [pos_c.x0, pos_m_cl_last.y0, pos_c.width, pos_m_cl_last.height]
        )
        pos_t = ax_tm.get_position(original=False)
        ax_tm.set_position(
            [pos_t.x0, pos_m_cl_last.y0, pos_t.width, pos_m_cl_last.height]
        )

    # -------------------------------------------------------------------------
    # Panel C: 4 Motorized Treadmill Examples (ROIs 49, 249, 41, 361)
    # -------------------------------------------------------------------------
    _panel_header(
        fig,
        y_head2,
        "C",
        "Motorized treadmill, trial-average plateau",
    )

    legend_tm = [
        Line2D(
            [0],
            [0],
            color=color_tm,
            ls="",
            marker="o",
            markersize=2.6,
            label="Stimulus",
        ),
        Line2D([0], [0], color=color_tm, lw=1.4, ls="-", label="1D RS fit"),
        Line2D([0], [0], color="#7F7F7F", lw=1.0, ls="--", label="Blank"),
    ]
    fig.legend(
        handles=legend_tm,
        loc="upper right",
        bbox_to_anchor=(0.95, y_head2 + 0.005),
        ncol=3,
        frameon=False,
        fontsize=FONTSIZE_DICT["legend"] - 1.5,
        handlelength=1.1,
        columnspacing=0.8,
    )

    # 4 depth-tuned treadmill example neurons spanning preferred speeds
    tm_examples = [
        ("PZAG17.3a_S20250402", 49, "PZAG17.3a"),
        ("PZAG17.3a_S20250402", 249, "PZAG17.3a"),
        ("PZAG16.3c_S20250401", 41, "PZAG16.3c"),
        ("PZAG16.3b_S20250401", 361, "PZAG16.3b"),
    ]

    speed_bins_tm = np.array([0, 5, 11, 22, 45, 90])
    bin_centers_tm = np.array([3.8, 7.6, 15.2, 30.5, 60.8])
    smooth_rs_grid = np.linspace(2, 70, 100)
    smooth_log_rs = np.log(smooth_rs_grid / 100)

    tm_sessions_data = {EXAMPLE_SESSION_TM: (trials_df_tm, red_tm)}

    for col_idx, (sess_name, roi, mouse_label) in enumerate(tm_examples):
        gs_cell = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs[1, col_idx], width_ratios=[1.3, 1.0], wspace=0.42
        )
        ax = fig.add_subplot(gs_cell[0, 0])
        ax_m = fig.add_subplot(gs_cell[0, 1])

        if sess_name not in tm_sessions_data:
            if flm_tm is None:
                flm_tm = flz.get_flexilims_session(PROJECT_TM)
            tm_sessions_data[sess_name] = load_tm_example_data(sess_name, flm_tm)

        t_df, r_data = tm_sessions_data[sess_name]

        cell_row = df_cells_tm[
            (df_cells_tm["session"] == sess_name) & (df_cells_tm["roi"] == roi)
        ].iloc[0]
        popt_grs = cell_row[f"rsof_popt_closedloop_grs{TA}"]
        rsq_grs = cell_row[f"rsof_test_rsq_closedloop_grs{TA}"]
        pref_grs = cell_row["preferred_RS_grs_cm_s"]

        rs_tm_trials = r_data["rs"] * 100  # cm/s
        dff_tm_trials = r_data["dff"]  # (n_trials, n_neurons)
        rs_idx = np.digitize(rs_tm_trials, speed_bins_tm) - 1

        means_stim, sems_stim = [], []
        for b_idx in range(len(bin_centers_tm)):
            vals_all = dff_tm_trials[rs_idx == b_idx, roi]
            means_stim.append(vals_all.mean())
            sems_stim.append(vals_all.std() / np.sqrt(len(vals_all)))

        ax.errorbar(
            bin_centers_tm,
            np.array(means_stim),
            yerr=np.array(sems_stim),
            fmt="o",
            color=color_tm,
            ms=2.3,
            capsize=1.0,
            linewidth=0.6,
            zorder=4,
        )

        if popt_grs is not None and len(popt_grs) >= 4:
            fit_grs_tm = fit_gaussian_blob.gaussian_1d(
                smooth_log_rs, *popt_grs, min_sigma=0.25
            )
            ax.plot(
                smooth_rs_grid, fit_grs_tm, color=color_tm, lw=1.3, ls="-", zorder=3
            )

        cl_tm_trials = t_df[t_df.closed_loop == 1]
        blank_vals = [
            np.nanmean(d[:, roi]) for d in cl_tm_trials["dff_blank"] if len(d)
        ]
        if len(blank_vals):
            ax.axhline(
                np.nanmean(blank_vals), color="#7F7F7F", lw=1.0, ls="--", zorder=1
            )

        ax.set_xlim(-2, 75)
        ax.set_xticks([0, 30, 60])
        ax.tick_params(axis="both", labelsize=FONTSIZE_DICT["tick"] - 1.5)
        sns.despine(ax=ax, offset=2, trim=True)
        ax.set_xlabel("Speed (cm/s)", fontsize=FONTSIZE_DICT["label"] - 1)
        if col_idx == 0:
            ax.set_ylabel(r"$\Delta$F/F", fontsize=FONTSIZE_DICT["label"])

        rs_bins_this, of_bins_this, tick_dict_this = compute_treadmill_rsof_bins(t_df)
        tm_range_kwargs = dict(
            log_range={"log_base": 2},
            rs_bins=rs_bins_this,
            of_bins=of_bins_this,
            tick_dict=tick_dict_this,
        )

        pos_m = _plot_summary_matrix(
            ax_m,
            t_df,
            roi,
            xlabel="RS (cm/s)",
            ylabel="OF (°/s)" if col_idx == 0 else "",
            is_tm=True,
            **tm_range_kwargs,
        )
        pos_ax = ax.get_position(original=False)
        ax.set_position([pos_ax.x0, pos_m.y0, pos_ax.width, pos_m.height])
        _cell_title(
            fig,
            gs,
            1,
            col_idx,
            f"ROI {roi} ({mouse_label})\n$r_0$={_fmt_speed(pref_grs)} cm/s, $R^2$={rsq_grs:.2f}",
            y=pos_m.y1 + 0.005,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "running_tuning_depth_pref_summary.png"
    out_pdf = out_dir / "running_tuning_depth_pref_summary.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf, transparent=True)
    print(f"Saved depth pref summary figure to:\n  {out_png}\n  {out_pdf}")
    return fig


def main():
    flm_cl = flz.get_flexilims_session(PROJECT_CL)
    flm_tm = flz.get_flexilims_session(PROJECT_TM)

    df_cells_cl = load_closedloop_population_data(flm_cl)
    trials_df_cl = load_cl_example_trials_df(EXAMPLE_SESSION_CL, flm_cl)

    df_cells_tm = load_treadmill_population_data(flm_tm)
    trials_df_tm, red_tm = load_tm_example_data(EXAMPLE_SESSION_TM, flm_tm)

    out_dir = get_output_dir(flm_cl)

    generate_depth_pref_summary_figure(
        df_cells_cl,
        trials_df_cl,
        df_cells_tm,
        trials_df_tm,
        red_tm,
        out_dir,
        flm_tm=flm_tm,
    )
    generate_figure(
        df_cells_cl, trials_df_cl, df_cells_tm, trials_df_tm, red_tm, out_dir
    )
    # Middle column of panel B
    generate_rs_scatter_figure(
        df_cells_tm, trials_df_tm, TM_RS_EXAMPLE_ROIS[1::3], out_dir
    )
    generate_pref_rs_histogram(df_cells_cl, df_cells_tm, out_dir)
    print("All done!")


if __name__ == "__main__":
    main()
