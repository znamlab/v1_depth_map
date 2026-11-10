"""Fig 3 G-I: preferred running speed (RS) vs optic flow (OF) on the motorized wheel, with
example neurons.

- G: preferred RS vs preferred OF (trial-average plateau 2D Gaussian fit, g2d) of the
  depth-selective neurons with a significant wheel RS/OF fit (the panel F population),
  coloured by preferred depth in free locomotion; marginal histograms of the preferences
  (RS above, OF on the right).
- H: depth tuning of the example neurons: motorized wheel (black, per-trial means and the
  1D Gaussian depth fit) and free locomotion (coloured, mean and bootstrapped 95% CI at
  the 8 depths).
- I: RS x OF matrix of each example (open circle: g2d preference) and the OF tuning of one
  example within each motor speed bin (as the old panel I).

Data are prepared by `prepare_example_data` (heavy: one session load per example session)
and the block is drawn by `plot_treadmill_examples_block` in cm coordinates on an
existing figure.

The wheel depth fits come from the refit with the fixed best-depth seeding
(`precompute_data/refit_treadmill_depth.py`, results under
`<processed>/colasa_3d-vision_revisions/analysis/treadmill_depth_refit/results`): the
fits stored in neurons_df were seeded at a NaN-argmax depth and many are flat.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from cottage_analysis.analysis import (
    common_utils,
    find_depth_neurons,
    fit_gaussian_blob,
)
from cottage_analysis.plotting import rsof_plots
from cottage_analysis.plotting.style import FONTSIZE_DICT
from v1_depth_map.figure_utils.treadmill import compute_treadmill_rsof_bins

TM_MOTOR_SPEEDS = np.array([3.81, 7.62, 15.25, 30.5, 61.0])  # cm/s
TM_OF_LEVELS = np.array([1, 4, 16, 64, 256, 1024])  # deg/s
# depths tested on the wheel: RS / OF over the motor speed x OF grid (cm)
GRID_DEPTHS_CM = TM_MOTOR_SPEEDS[None, :] / np.radians(TM_OF_LEVELS)[:, None]
MAX_RS2MOTOR_DIFF = 0.3  # as the depth and RS/OF fits
DEPTH_RS_THR = 0  # fit_preferred_depth default: frames with RS > 0
DEPTH_MIN_SIGMA = 0.5
TM_DEPTH_SFX = "_treadmill"  # refit variant (mirrored to _treadmill_plateau)
REFIT_SUBDIR = "analysis/treadmill_depth_refit/results"
CMAP = "cool_r"

# Layout (cm). x from the block left, y from the block top.
LAYOUT = dict(
    left=1.0,  # scatter y label + ticks
    top=0.6,  # legend above the depth grid
    bottom=0.95,  # x tick labels + labels
    cbar_gap=0.2,
    cbar_w=0.108,
    sc_gap=2.0,  # colour bar ticks / title, then example y label + ticks
    ex_w=1.6,  # example panel
    ex_h=1.4,
    ex_hgap=0.48,  # between the 2 columns of a grid
    ex_vgap=0.4,  # between example rows
    grid_gap=0.9,  # between grids
    mat_cbar_gap=0.06,
    mat_cbar_w=0.06,
    mat_hgap=0.75,  # matrix to next matrix column (colour bar + its tick labels)
    mat_vgap=0.12,  # between matrix rows
    mat_h=1.65,  # matrix height; the stack is centred on the depth grid
)
MAT_CBAR_FRAC = 0.5  # matrix colour bar height relative to the matrix (bottom-aligned)
CBAR_FRAC = 0.36  # colour bar height relative to the scatter
MAT_RS_OCT, MAT_OF_OCT = 5, 12  # matrix extent in octaves (motor speed, OF)
MARG_H = 0.9  # thickness of the marginal histograms
MARG_GAP = 0.12  # scatter to marginal
HIST_W_FRAC = 1.3  # OF-by-RS column width relative to an example panel
BAR_HALF_IN = 0.45  # in-range bar half-width, in steps between sampled values (log)
BAR_HALF_OUT = 0.15  # out-of-range bar half-width
BAR_GAP = 0.08  # minimum gap next to the out-of-range bars, in steps
LIM_PAD = 1.2  # axis limit beyond the most extreme value (room for a dot radius)


# ---------------------------------------------------------------------------- data
def load_treadmill_depth_refit(processed_root, session, project):
    """Wheel depth fit columns (refit with the fixed seed) of one session, indexed by roi."""
    path = (
        processed_root
        / project
        / REFIT_SUBDIR
        / session
        / f"treadmill_depth_fit{TM_DEPTH_SFX}.pickle"
    )
    return pd.read_pickle(path).set_index("roi")


def depth_trial_means(trials_df, closed_loop_only=True, rs_thr=None, rs2motor=True):
    """Per-trial mean dF/F over the stimulus frames (the depth fit input).

    Returns depth (cm, per trial) and dF/F (trials, rois).
    """
    t = trials_df
    if rs2motor:
        t = common_utils.filter_trials_by_rs2motor(
            t, max_rs2motor_diff=MAX_RS2MOTOR_DIFF
        )
    t = t[~t.recording_name.str.contains("multidepth")]
    if closed_loop_only:
        t = t[t.closed_loop == 1]
    kwargs = {} if rs_thr is None else dict(rs_thr_min=rs_thr)
    t = find_depth_neurons.trial_average_dff(t.copy(), **kwargs)
    return t["depth"].values * 100, np.stack(t["trial_mean_dff"].values)


def _one_roi_trials(trials_df, roi):
    """Trials with dF/F reduced to one ROI (column 0), for the matrix and OF slices."""
    keep = [
        c for c in trials_df.columns if not (c.startswith("dff_") and c != "dff_stim")
    ]
    t = trials_df[keep].copy()
    t["dff_stim"] = [d[:, [roi]] for d in trials_df["dff_stim"]]
    return t


def prepare_example_data(examples, trials_by_session, refits):
    """Small per-example data for the H/I panels (cache it: it needs the full sessions).

    Args:
        examples: list of roi_uid strings ("MOUSE_SESSION_ROI").
        trials_by_session: {session: (trials_df_treadmill, trials_df_sphere)}, wheel
            trials loaded with `tread_kwargs=dict(method="plateau")`.
        refits: {session: refit dataframe} from `load_treadmill_depth_refit`.

    Returns:
        list of dicts, one per example, in order.
    """
    out, means = [], {}
    for uid in examples:
        session, roi = uid.rsplit("_", 1)
        roi = int(roi)
        if session not in means:
            tm, sphere = trials_by_session[session]
            means[session] = (
                depth_trial_means(tm, rs_thr=DEPTH_RS_THR),
                # free locomotion: all stimulus frames of closed-loop trials, as the
                # closed-loop depth fit (rs_thr=None)
                depth_trial_means(sphere, rs2motor=False),
            )
        (tm_depth, tm_dff), (cl_depth, cl_dff) = means[session]
        refit = refits[session].loc[roi]
        out.append(
            dict(
                roi_uid=uid,
                session=session,
                roi=roi,
                tm_depth_cm=tm_depth,
                tm_dff=tm_dff[:, roi],
                cl_depth_cm=cl_depth,
                cl_dff=cl_dff[:, roi],
                tm_depth_popt=np.asarray(
                    refit[f"depth_tuning_popt_closedloop{TM_DEPTH_SFX}"], float
                ),
                tm_pref_depth_cm=float(
                    refit[f"preferred_depth_closedloop{TM_DEPTH_SFX}"] * 100
                ),
                trials=_one_roi_trials(trials_by_session[session][0], roi),
            )
        )
    return out


def preference_table(neurons_df, ta_suffix):
    """Preferred RS (cm/s), OF (deg/s) on the wheel (g2d) and free-locomotion depth (cm)."""
    return pd.DataFrame(
        {
            "roi_uid": neurons_df["roi_uid"],
            "pref_rs": neurons_df[f"preferred_RS_closedloop_g2d{ta_suffix}"] * 100,
            "pref_of": np.degrees(
                neurons_df[f"preferred_OF_closedloop_g2d{ta_suffix}"]
            ),
            "pref_depth_cl": neurons_df["preferred_depth_closedloop"] * 100,
        },
        index=neurons_df.index,
    )


# --------------------------------------------------------------------------- panels
def marginal_bars(levels, values):
    """Bar geometry of a marginal on a log axis with sampled `levels`.

    In-range bars are centred on the levels and nearly touch. Values below / above the
    range are fits at the fit bound: their bar is centred on the most extreme value
    (where they pile up on the scatter) and narrower, and the end in-range bar is trimmed
    on its outer side to leave a gap. Returns (centres, lefts, rights, lim) for the bars
    [levels..., below, above] and the axis limits.
    """
    lv = np.sort(np.asarray(levels, float))
    v = np.asarray(values, float)
    step = np.exp(np.median(np.diff(np.log(lv))))
    half_in, half_out, gap = step**BAR_HALF_IN, step**BAR_HALF_OUT, step**BAR_GAP
    low_pos = v.min() if (v < lv[0]).any() else lv[0] / step
    high_pos = v.max() if (v > lv[-1]).any() else lv[-1] * step
    centres = np.r_[lv, low_pos, high_pos]
    lefts = np.r_[lv / half_in, low_pos / half_out, high_pos / half_out]
    rights = np.r_[lv * half_in, low_pos * half_out, high_pos * half_out]
    lefts[0] = max(lefts[0], rights[-2] * gap)  # trim the end bars on their outer side
    rights[len(lv) - 1] = min(rights[len(lv) - 1], lefts[-1] / gap)
    lim = (min(lefts.min(), v.min() / LIM_PAD), max(rights.max(), v.max() * LIM_PAD))
    return centres, lefts, rights, lim


def plot_marginal(ax, values, levels, lim, orientation, fontsize_dict=FONTSIZE_DICT):
    """Marginal histogram of the scatter on its log axis: one wide bar per sampled value
    (values within the sampled range go to the nearest one, in log); values below / above
    the range (fits at the fit bounds) are grouped in one narrower bar each."""
    v = np.asarray(values, float)
    lv = np.sort(np.asarray(levels, float))
    lo, hi = lv[0], lv[-1]
    _, lefts, rights, _ = marginal_bars(lv, v)
    inside = (v >= lo) & (v <= hi)
    idx = np.abs(np.log(v[inside])[:, None] - np.log(lv)).argmin(1)
    counts = np.r_[np.bincount(idx, minlength=len(lv)), (v < lo).sum(), (v > hi).sum()]
    top = int(np.ceil(counts.max() / 10) * 10)  # first multiple of 10 above the max
    if orientation == "vertical":  # above the scatter, shares its x axis
        ax.bar(lefts, counts, width=rights - lefts, align="edge", color="0.6", lw=0)
        ax.set_xscale("log")
        ax.set_xlim(lim)
        ax.set_ylim(0, top)
        ax.xaxis.set_visible(False)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        count_axis = ax.yaxis
    else:  # right of the scatter, shares its y axis; count axis on top
        ax.barh(lefts, counts, height=rights - lefts, align="edge", color="0.6", lw=0)
        ax.set_yscale("log")
        ax.set_ylim(lim)
        ax.set_xlim(0, top)
        ax.yaxis.set_visible(False)
        ax.spines[["bottom", "right", "left"]].set_visible(False)
        ax.spines["top"].set_position(("outward", 4))  # clear of the top bar
        count_axis = ax.xaxis
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
    centres = np.sqrt(lefts * rights)  # bar centres on the log axis
    for centre, n in zip(centres, counts):  # number of cells at the end of each bar
        vertical = orientation == "vertical"
        ax.annotate(
            str(n),
            (centre, n) if vertical else (n, centre),
            xytext=(0, 1) if vertical else (1, 0),
            textcoords="offset points",
            rotation=90 if vertical else 0,
            ha="center" if vertical else "left",
            va="bottom" if vertical else "center",
            fontsize=fontsize_dict["tick"],
            annotation_clip=False,
        )
    count_axis.set_ticks([0, top])
    ax.tick_params(labelsize=fontsize_dict["tick"], length=1.5, pad=1)
    count_axis.set_label_text("Cell #", fontsize=fontsize_dict["tick"])
    count_axis.labelpad = 1
    return counts


def plot_example_depth(ax, ex, color, pref_depth_cl, fontsize_dict=FONTSIZE_DICT):
    """Depth tuning of one example: wheel (black, mean and bootstrapped 95% CI per depth,
    and the fit) and free locomotion (`color`, mean and bootstrapped 95% CI per depth);
    dashed lines at both preferences.

    x: the depths tested on the wheel (spine from the nearest to the farthest). y: from 0
    to a round top tick just above the highest point (CI upper bound or fit), at most 10%
    of the axis below it. Spines are offset here: do not trim them afterwards."""
    d_lo, d_hi = GRID_DEPTHS_CM.min(), GRID_DEPTHS_CM.max()
    x_fit = np.geomspace(d_lo, d_hi, 200)
    y_fit = fit_gaussian_blob.gaussian_1d(
        np.log(x_fit / 100), *ex["tm_depth_popt"], min_sigma=DEPTH_MIN_SIGMA
    )
    y_hi = [np.nanmax(y_fit)]
    for depth, dff, c, z in (
        (ex["tm_depth_cm"], ex["tm_dff"], "k", 3),
        (ex["cl_depth_cm"], ex["cl_dff"], color, 4),
    ):
        resp = pd.Series(dff).groupby(np.round(depth, 2))
        mean = resp.mean()
        ci = np.array(
            [
                [b[0] for b in common_utils.get_bootstrap_ci(g.to_numpy())]
                for _, g in resp
            ]
        )  # (n_depths, 2): low, high
        ax.errorbar(
            mean.index.values,
            mean.values,
            yerr=(mean.values - ci[:, 0], ci[:, 1] - mean.values),
            fmt="o",
            color=c,
            ms=1.5,
            lw=0.5,
            capsize=0,
            zorder=z,
        )
        y_hi.append(np.nanmax(np.fmax(mean.values, ci[:, 1])))
    ax.plot(x_fit, y_fit, color="k", lw=1)
    ax.set_xlim(d_lo / 1.3, d_hi * 1.3)
    for pref, c in ((ex["tm_pref_depth_cm"], "k"), (pref_depth_cl, color)):
        if x_fit[0] <= pref <= x_fit[-1]:
            ax.axvline(pref, color=c, lw=0.5, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000], ["1", "", "100", ""])
    ax.minorticks_off()
    ax.tick_params(labelsize=fontsize_dict["tick"], length=1.5, pad=1)
    y_max = max(y_hi)
    top = _round_top(y_max)
    ax.set_ylim(min(0, ax.get_ylim()[0]), max(top, y_max) * 1.02)
    ax.set_yticks([0, top], ["0", f"{top:g}"])
    sns.despine(ax=ax, offset=2)
    ax.spines["bottom"].set_bounds(d_lo, d_hi)
    ax.spines["left"].set_bounds(0, top)


def _round_top(y_max):
    """Round tick close to y_max, at most 10% away: the coarsest step (1, 0.5, 0.2, 0.1
    x the order of magnitude of y_max) giving such a tick, preferring a tick just above
    y_max to one just below."""
    order = 10 ** np.floor(np.log10(y_max))
    for step in order * np.array([1, 0.5, 0.2, 0.1]):
        above = np.ceil(y_max / step) * step
        if above <= y_max * 1.1:
            return float(np.round(above, 10))
        below = np.floor(y_max / step) * step
        if below >= y_max / 1.1:
            return float(np.round(below, 10))
    return float(np.round(np.ceil(y_max / (order / 10)) * order / 10, 10))


def _matrix_bins(trials_df):
    """Motor speed / OF bins without the column below the slowest motor speed."""
    rs_bins, of_bins, tick_dict = compute_treadmill_rsof_bins(trials_df)
    tick_dict = dict(
        tick_dict,
        rs_tick_select=tick_dict["rs_tick_select"][1:],
        rs_tick_values=tick_dict["rs_tick_values"][1:],
    )
    return np.delete(rs_bins, 1), of_bins, tick_dict


def plot_example_matrix(
    fig, ax, cax, ex, pref_rs, pref_of, show_xticks, fontsize_dict=FONTSIZE_DICT
):
    """RS x OF matrix (stimulus frames, 0 to the cell max) with its colour bar and an
    open circle at the g2d preference (clipped to the matrix)."""
    rs_bins, of_bins, tick_dict = _matrix_bins(ex["trials"])
    _, _, mat = rsof_plots.plot_RS_OF_matrix(
        trials_df=ex["trials"],
        roi=0,
        is_closed_loop=1,
        vmin=0,
        vmax=1,
        xlabel="",
        ylabel="",
        cbar_width=None,
        fontsize_dict=fontsize_dict,
        ax=ax,
        return_matrix=True,
        log_range={"log_base": 2},
        rs_bins=rs_bins,
        of_bins=of_bins,
        tick_dict=tick_dict,
    )
    vmax = max(np.ceil(np.nanmax(mat) * 100) / 100, 0.01)
    ax.images[0].set_clim(0, vmax)
    ax.set_aspect("equal", adjustable="box", anchor="NW")  # equal length per octave
    ax.tick_params(labelsize=fontsize_dict["tick"], length=1.5, pad=1)
    ax.set_xticklabels(  # every other motor speed label, the axis is narrow
        [t.get_text() if k % 2 == 0 else "" for k, t in enumerate(ax.get_xticklabels())]
    )
    if not show_xticks:
        ax.tick_params(labelbottom=False)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.scatter(
        np.clip(np.log2(pref_rs), *xlim),
        np.clip(np.log2(pref_of), *ylim),
        s=10,
        facecolor="none",
        edgecolor="w",
        linewidths=0.5,
        clip_on=False,
        zorder=5,
    )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cbar.set_ticks([0, vmax], labels=["0", f"{vmax:g}"])
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"], length=1, pad=1)
    cbar.outline.set_linewidth(0.5)


def plot_of_by_rs(add_axes_cm, x, y, w, h, ex, color="k", fontsize_dict=FONTSIZE_DICT):
    """OF tuning within each motor speed bin, stacked fastest (top) to slowest: trial
    averages over the frames passing the RS-to-motor check, mean +/- bootstrapped 95% CI,
    1D Gaussian fit, triangle at its mean just above the highest error bar. Each panel
    has its own dF/F axis; OF ticks at every tested level, labels on the bottom panel."""
    trials_df = ex["trials"]
    tav_df = []
    for _, tr in trials_df.iterrows():
        ok = tr.max_abs_rs2motor_diff_ratio_stim < MAX_RS2MOTOR_DIFF
        tav_df.append(
            dict(
                rs=tr.RS_stim[ok].mean() * 100,
                of=np.degrees(tr.OF_stim[ok].mean()),
                dff=tr.dff_stim[:, 0].mean(),
            )
        )
    tav_df = pd.DataFrame(tav_df)
    rs_bins, of_bins, _ = compute_treadmill_rsof_bins(trials_df)
    n_bins, gap = len(rs_bins) - 3, 0.3  # gap: room for the fit-mean triangle
    stack_h = (h - (n_bins - 1) * gap) / n_bins
    axes = [add_axes_cm(x, y + i * (stack_h + gap), w, stack_h) for i in range(n_bins)]
    for b_s, b_e, a in zip(rs_bins[2:], rs_bins[3:], axes[::-1]):  # fastest on top
        rsof_plots.plot_rsof_slice(
            a,
            b_s,
            b_e,
            tav_df,
            of_bins[1:],
            plot_trials=False,
            niter=10,
            color=color,
            fit_color=color,
            scatter_size=20,
            linewidth=1,
            capsize=1.5,
            markersize=3,
            fontsize_dict=fontsize_dict,
            clip_on=False,
        )
        for line in list(a.lines):  # drop the grey y=0 axhline drawn by plot_rsof_slice
            if list(line.get_xdata()) == [0, 1] and list(line.get_ydata()) == [0, 0]:
                line.remove()
        a.set_xscale("log")

    def data_range(ax):
        """Lowest / highest point of the error bars and the fit curve."""
        lo, hi = [0.0], []
        for container in ax.containers:
            line, _, bars = container.lines
            hi.append(np.nanmax(line.get_ydata()))
            for bar in bars:
                for seg in bar.get_segments():
                    lo.append(np.nanmin(seg[:, 1]))
                    hi.append(np.nanmax(seg[:, 1]))
        for line in ax.lines:
            yd = np.asarray(line.get_ydata(), float)
            if yd.size > 100:  # the fit curve
                lo.append(np.nanmin(yd))
                hi.append(np.nanmax(yd))
        return min(lo), max(hi)

    for i, a in enumerate(axes):
        lo, hi = data_range(a)
        span = hi - lo
        for line in list(a.lines):  # fit-mean triangle just above the highest error bar
            if line.get_marker() == "v":
                of_mean = line.get_xdata()[0]
                line.remove()
                a.plot(
                    of_mean,
                    hi + 0.12 * span,
                    marker="v",
                    color=color,
                    markersize=3,
                    clip_on=False,
                )
        a.set_ylim(lo - 0.05 * span, hi + 0.2 * span)
        a.yaxis.set_major_locator(MaxNLocator(nbins=2, min_n_ticks=2))
        a.tick_params(labelsize=fontsize_dict["tick"], length=1.5, pad=1)
        a.set_ylabel("")
        a.spines[["top", "right"]].set_visible(False)
        a.set_xticks(
            TM_OF_LEVELS,
            labels=[
                f"{v:g}" if k % 2 == 0 and i == n_bins - 1 else ""
                for k, v in enumerate(TM_OF_LEVELS)
            ],
        )
        a.minorticks_off()
    axes[n_bins // 2].set_ylabel("ΔF/F", fontsize=fontsize_dict["label"], labelpad=1)
    axes[-1].set_xlabel("Optic flow (°/s)", fontsize=fontsize_dict["label"], labelpad=1)
    return axes


# ---------------------------------------------------------------------------- block
def block_size(scatter_df, n_rows=3, n_cols=2):
    """(width, height) of the block in cm, margins included."""
    geom = _geometry(scatter_df, n_rows, n_cols)
    return geom["right"], LAYOUT["top"] + geom["grid_h"] + LAYOUT["bottom"]


def _geometry(scatter_df, n_rows, n_cols):
    L = LAYOUT
    grid_h = n_rows * L["ex_h"] + (n_rows - 1) * L["ex_vgap"]
    grid_w = n_cols * L["ex_w"] + (n_cols - 1) * L["ex_hgap"]
    rs_lim = marginal_bars(TM_MOTOR_SPEEDS, scatter_df["pref_rs"])[3]
    of_lim = marginal_bars(TM_OF_LEVELS, scatter_df["pref_of"])[3]
    # matrices: x tick labels only on the bottom row, cell names inside, equal length
    # per octave on both axes; the stack is centred vertically on the depth grid
    mat_h = L["mat_h"]
    mat_y0 = (grid_h - n_rows * mat_h - (n_rows - 1) * L["mat_vgap"]) / 2
    mat_w = mat_h * MAT_RS_OCT / MAT_OF_OCT
    mat_grid_w = n_cols * mat_w + (n_cols - 1) * L["mat_hgap"]
    marg_h = MARG_H + MARG_GAP
    scat_h = grid_h - marg_h  # scatter + top marginal span the example grid height
    sc_w = scat_h * np.log(rs_lim[1] / rs_lim[0]) / np.log(of_lim[1] / of_lim[0])
    cbar_x = L["left"] + sc_w + marg_h + L["cbar_gap"]
    x_depth = cbar_x + L["cbar_w"] + L["sc_gap"]
    x_mat = x_depth + grid_w + L["grid_gap"] + 0.65  # matrix OF label + tick labels
    x_ofrs = x_mat + mat_grid_w + L["grid_gap"] + 0.45
    # OF-by-RS column takes the width the matrices gave back (block width unchanged)
    old_mat_grid_w = n_cols * L["ex_h"] * MAT_RS_OCT / MAT_OF_OCT + (n_cols - 1) * 1.3
    hist_w = L["ex_w"] * HIST_W_FRAC + old_mat_grid_w - mat_grid_w
    return dict(
        grid_h=grid_h,
        grid_w=grid_w,
        rs_lim=rs_lim,
        of_lim=of_lim,
        mat_w=mat_w,
        mat_h=mat_h,
        mat_y0=mat_y0,
        mat_grid_w=mat_grid_w,
        marg_h=marg_h,
        scat_h=scat_h,
        sc_w=sc_w,
        cbar_x=cbar_x,
        x_depth=x_depth,
        x_mat=x_mat,
        x_ofrs=x_ofrs,
        hist_w=hist_w,
        right=x_ofrs + hist_w,
    )


def plot_treadmill_examples_block(
    fig,
    fig_w,
    fig_h,
    x0,
    y0,
    scatter_df,
    example_data,
    example_prefs,
    of_by_rs_cell,
    n_rows=3,
    n_cols=2,
    fontsize_dict=FONTSIZE_DICT,
):
    """Draw G-I on `fig` (size fig_w x fig_h cm) with the block's top-left at (x0, y0) cm.

    Args:
        scatter_df: `preference_table` rows of the scatter population.
        example_data: `prepare_example_data` output, grid in reading order.
        example_prefs: `preference_table` rows of the examples, same order.
        of_by_rs_cell: 1-based example shown with its OF tuning per motor speed.

    Returns:
        dict of the x positions (cm, figure) of the scatter, depth grid and matrix grid,
        for the panel letters.
    """
    L = LAYOUT
    g = _geometry(scatter_df, n_rows, n_cols)

    def add_axes_cm(x, y, w, h):
        return fig.add_axes(
            [(x0 + x) / fig_w, 1 - (y0 + y + h) / fig_h, w / fig_w, h / fig_h]
        )

    def text_cm(x, y, s, **kw):
        return fig.text((x0 + x) / fig_w, 1 - (y0 + y) / fig_h, s, **kw)

    cmap = plt.get_cmap(CMAP)
    depth_cl = scatter_df["pref_depth_cl"]
    norm = LogNorm(depth_cl.min(), depth_cl.max())
    y_top = L["top"]  # top of the axes
    scat_y = y_top + g["marg_h"]

    # --- G: scatter, equal length per log unit on both axes
    ax = add_axes_cm(L["left"], scat_y, g["sc_w"], g["scat_h"])
    d = scatter_df.iloc[np.random.default_rng(0).permutation(len(scatter_df))]
    sc = ax.scatter(
        d["pref_rs"],
        d["pref_of"],
        c=d["pref_depth_cl"],
        cmap=cmap,
        norm=norm,
        s=8,
        alpha=0.6,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*g["rs_lim"])
    ax.set_ylim(*g["of_lim"])
    ax.set_xticks(TM_MOTOR_SPEEDS, [f"{v:.0f}" for v in TM_MOTOR_SPEEDS])
    ax.set_yticks(TM_OF_LEVELS, [f"{v:g}" for v in TM_OF_LEVELS])
    ax.minorticks_off()
    ax.tick_params(labelsize=fontsize_dict["tick"], length=1.5, pad=1)
    ax.set_xlabel(
        "Preferred running speed (cm/s)", fontsize=fontsize_dict["label"], labelpad=1
    )
    ax.set_ylabel(
        "Preferred optic flow (°/s)", fontsize=fontsize_dict["label"], labelpad=1
    )
    sns.despine(ax=ax, offset=2, trim=True)
    plot_marginal(
        add_axes_cm(L["left"], y_top, g["sc_w"], MARG_H),
        scatter_df["pref_rs"],
        TM_MOTOR_SPEEDS,
        g["rs_lim"],
        "vertical",
        fontsize_dict,
    )
    plot_marginal(
        add_axes_cm(L["left"] + g["sc_w"] + MARG_GAP, scat_y, MARG_H, g["scat_h"]),
        scatter_df["pref_of"],
        TM_OF_LEVELS,
        g["of_lim"],
        "horizontal",
        fontsize_dict,
    )
    for i, (_, row) in enumerate(example_prefs.iterrows()):
        ax.scatter(
            row["pref_rs"],
            row["pref_of"],
            s=12,
            facecolor="none",
            edgecolor="k",
            linewidths=0.5,
            zorder=5,
        )
        close = any(  # label below-right if an earlier example is within ~1/3 octave
            np.hypot(
                np.log2(row["pref_rs"] / r2["pref_rs"]),
                np.log2(row["pref_of"] / r2["pref_of"]),
            )
            < 0.35
            for _, r2 in example_prefs.iloc[:i].iterrows()
        )
        ax.annotate(
            str(i + 1),
            (row["pref_rs"], row["pref_of"]),
            xytext=(2, -6) if close else (2, 2),
            textcoords="offset points",
            fontsize=fontsize_dict["tick"],
        )
    # colour bar, bottom-aligned with the scatter
    cbar_h = g["scat_h"] * CBAR_FRAC
    cax = add_axes_cm(g["cbar_x"], scat_y + g["scat_h"] - cbar_h, L["cbar_w"], cbar_h)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.solids.set_alpha(1)
    cbar.ax.set_title(
        "Preferred\ndepth, free\nlocomotion (cm)",
        fontsize=fontsize_dict["legend"],
        ha="left",
        pad=3,
    )
    cbar_ticks = [t for t in (1, 10, 100, 1000) if norm.vmin <= t <= norm.vmax]
    cbar.set_ticks(cbar_ticks, labels=[f"{t:g}" for t in cbar_ticks])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=fontsize_dict["tick"], length=1.5, pad=1)
    cbar.outline.set_linewidth(0.5)

    # --- H: depth tuning grid, legend above
    fig.legend(
        handles=[
            Line2D([], [], color="k", marker="o", ms=2, lw=1, label="Motorized wheel"),
            Line2D(
                [], [], color=cmap(0.5), marker="o", ms=2, lw=0, label="Free locomotion"
            ),
        ],
        loc="lower left",
        bbox_to_anchor=((x0 + g["x_depth"]) / fig_w, 1 - (y0 + y_top - 0.3) / fig_h),
        ncol=2,
        frameon=False,
        fontsize=fontsize_dict["legend"],
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.0,
        borderpad=0,
        borderaxespad=0,
    )
    xlabel_y = y_top + g["grid_h"] + 0.38
    text_cm(
        g["x_depth"] + g["grid_w"] / 2,
        xlabel_y,
        "Depth (cm)",
        ha="center",
        va="top",
        fontsize=fontsize_dict["label"],
    )
    # --- I: matrices, then the OF tuning per motor speed of one example
    text_cm(
        g["x_mat"] - 0.62,
        y_top + g["grid_h"] / 2,
        "Optic flow (°/s)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=fontsize_dict["label"],
    )
    text_cm(
        g["x_mat"] + g["mat_grid_w"] / 2,
        xlabel_y,
        "Running speed (cm/s)",
        ha="center",
        va="top",
        fontsize=fontsize_dict["label"],
    )
    for i, (ex, (_, pref)) in enumerate(zip(example_data, example_prefs.iterrows())):
        r, c = divmod(i, n_cols)
        y = y_top + r * (L["ex_h"] + L["ex_vgap"])
        color = cmap(norm(pref["pref_depth_cl"]))
        x = g["x_depth"] + c * (L["ex_w"] + L["ex_hgap"])
        ax_ex = add_axes_cm(x, y, L["ex_w"], L["ex_h"])
        plot_example_depth(ax_ex, ex, color, pref["pref_depth_cl"], fontsize_dict)
        if r < n_rows - 1:
            ax_ex.tick_params(labelbottom=False)
        if c == 0:
            ax_ex.set_ylabel("ΔF/F", fontsize=fontsize_dict["label"], labelpad=1)
        xm = g["x_mat"] + c * (g["mat_w"] + L["mat_hgap"])
        ym = y_top + g["mat_y0"] + r * (g["mat_h"] + L["mat_vgap"])
        ax_m = add_axes_cm(xm, ym, g["mat_w"], g["mat_h"])
        cbar_h = g["mat_h"] * MAT_CBAR_FRAC
        cax_m = add_axes_cm(
            xm + g["mat_w"] + L["mat_cbar_gap"],
            ym + g["mat_h"] - cbar_h,
            L["mat_cbar_w"],
            cbar_h,
        )
        plot_example_matrix(
            fig,
            ax_m,
            cax_m,
            ex,
            pref["pref_rs"],
            pref["pref_of"],
            r == n_rows - 1,
            fontsize_dict,
        )
        if c > 0:  # OF tick labels on the first column only
            ax_m.tick_params(labelleft=False)
        text_cm(  # depth panel name, above its top-right corner
            x + L["ex_w"],
            y - 0.03,
            f"Cell {i + 1}",
            ha="right",
            va="bottom",
            fontsize=fontsize_dict["tick"],
        )
        ax_m.text(  # matrix name, inside its bottom-left corner (lowest OF: no circles)
            0.06,
            0.02,
            f"Cell {i + 1}",
            transform=ax_m.transAxes,
            ha="left",
            va="bottom",
            fontsize=fontsize_dict["tick"],
            path_effects=[patheffects.withStroke(linewidth=1.2, foreground="w")],
            zorder=6,
        )
    plot_of_by_rs(
        add_axes_cm,
        g["x_ofrs"],
        y_top + 0.3,  # room for the fit triangle
        g["hist_w"],
        g["grid_h"] - 0.3,
        example_data[of_by_rs_cell - 1],
        "k",
        fontsize_dict,
    )
    text_cm(
        g["x_ofrs"] + g["hist_w"],
        y_top - 0.03,
        f"Cell {of_by_rs_cell}, by motor speed",
        ha="right",
        va="bottom",
        fontsize=fontsize_dict["tick"],
    )
    return dict(
        x_scatter=x0,
        x_depth=x0 + g["x_depth"] - 1.0,
        x_matrix=x0 + g["x_mat"] - 1.0,
        x_of_by_rs=x0 + g["x_ofrs"] - 0.6,
    )
