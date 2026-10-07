"""Pairwise RF / depth-preference distance as a function of cortical distance.

Used for Figure 4 I-K: mean |delta(RF azimuth)|, |delta(RF elevation)| and
|delta log2(preferred virtual depth)| between depth-tuned neurons with
significant RFs, binned by the distance between their ROI centres, with
confidence bands from a hierarchical (mouse, then session) bootstrap.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import pairwise_distances

import flexiznam as flz
from cottage_analysis.analysis import common_utils, roi_location
from cottage_analysis.analysis.spheres import rf_analysis, rf_fitting
from cottage_analysis.pipelines import pipeline_utils

PIXEL_SIZE = 661 / 1024  # 2p FOV 661 um, 1024 pixels
CI_COLS = ["rf_azi", "rf_ele", "preferred_depth_closedloop"]
BINS = np.linspace(10, 860, 30)
YLABELS = [
    r"Mean |$\Delta$(RF azimuth)| (degrees)",
    r"Mean |$\Delta$(RF elevation)| (degrees)",
    r"Mean |$\Delta\log_2$(preferred virtual depth)|",
]


def load_session_neurons(flexilims_session, session_name, rf_file=None):
    """neurons_df of one V1 session with RF and ROI centres, and the selected neurons.

    Adds rf_azi, rf_ele, center_x and center_y (pixels), depth_tuned, rf_sig and
    preferred_depth_corrected. Selected neurons are depth-tuned cells with a
    significant RF.

    Returns:
        neurons_df (pd.DataFrame), select_neurons (boolean pd.Series)
    """
    # find_ndepths
    if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
        ndepths = 5
    else:
        ndepths = 8
    # load neurons_df and stat
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session_name,
        flexilims_session=flexilims_session,
        conflicts="skip",
    )

    # rf_file: RF columns from another fit, e.g. "neurons_df_spks.pickle"
    neurons_df = pipeline_utils.load_neurons_df(neurons_ds, rf_file=rf_file)

    suite2p_ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        allow_multiple=False,
        return_dataseries=False,
    )
    stat = np.load(suite2p_ds.path_full / "plane0" / "stat.npy", allow_pickle=True)
    iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[
        :, 0
    ]

    neurons_df["iscell"] = iscell
    common_utils.add_one_sided_spearman_significance(
        neurons_df,
        rval_col="depth_tuning_test_spearmanr_rval_closedloop",
        pval_col="depth_tuning_test_spearmanr_pval_closedloop",
        out_col="depth_tuned",
    )

    # find rf and roi centers
    rf_analysis.find_rf_centers(
        neurons_df,
        ndepths=ndepths,
        frame_shape=(16, 24),
        is_closed_loop=1,
        resolution=5,
    )
    roi_location.find_roi_centers(neurons_df, stat)

    # correct preferred depth
    neurons_df["preferred_depth_corrected"] = neurons_df[
        "preferred_depth_closedloop"
    ] / np.sqrt(
        (np.sin(np.deg2rad(neurons_df["rf_azi"])) ** 2)
        * (np.cos(np.deg2rad(neurons_df["rf_ele"])) ** 2)
        + (np.sin(np.deg2rad(neurons_df["rf_ele"])) ** 2)
    )
    # find significant rfs
    coef = np.stack(neurons_df["rf_coef_closedloop"].values)
    coef_ipsi = np.stack(neurons_df["rf_coef_ipsi_closedloop"].values)
    if coef_ipsi.ndim == 3:
        sig, sig_ipsi = rf_fitting.find_sig_rfs(
            np.swapaxes(np.swapaxes(coef, 0, 2), 0, 1),
            np.swapaxes(np.swapaxes(coef_ipsi, 0, 2), 0, 1),
            n_std=6,
        )
        neurons_df["rf_sig"] = sig
        neurons_df["rf_sig_ipsi"] = sig_ipsi

    neurons_df["preferred_depth_amplitude"] = neurons_df[
        "depth_tuning_popt_closedloop"
    ].apply(lambda x: np.exp(x[0]) + x[-1])
    select_neurons = (
        (neurons_df.depth_tuned) & (neurons_df.iscell) & (neurons_df.rf_sig == 1)
    )
    return neurons_df, select_neurons


# calculate pairwise distance for roi centers, rf azimuth, elevation, and depth
def calculate_pairwise_distance_per_session(
    flexilims_session,
    session_name,
    rf_file=None,
):
    neurons_df, select_neurons = load_session_neurons(
        flexilims_session, session_name, rf_file=rf_file
    )
    # find pairwise distance for roi coordinate centers
    session_df = pd.DataFrame()
    coords = [
        [i, j]
        for i, j in zip(
            neurons_df[select_neurons]["center_x"],
            neurons_df[select_neurons]["center_y"],
        )
    ]
    if len(coords) == 0:
        session_df["roi_distance"] = np.nan
        session_df["rf_azi_distance"] = np.nan
        session_df["rf_ele_distance"] = np.nan
        session_df["preferred_depth_closedloop_distance"] = np.nan
    else:
        ds = pairwise_distances(coords, metric="euclidean")
        ds = ds[np.triu_indices(ds.shape[0], k=1)]
        session_df["roi_distance"] = ds

        # find pairwise distance for rf azimuth, elevation, and depth
        for col in [
            "rf_azi",
            "rf_ele",
            "preferred_depth_closedloop",
            "preferred_depth_corrected",
        ]:
            if "preferred_depth" in col:
                coords = [[np.log2(i)] for i in neurons_df[select_neurons][col]]
            else:
                coords = [[i] for i in neurons_df[select_neurons][col]]
            col_ds = pairwise_distances(coords, metric="euclidean")
            col_ds = col_ds[np.triu_indices(col_ds.shape[0], k=1)]
            session_df[f"{col}_distance"] = col_ds

    session_df["session"] = session_name
    return session_df


def calculate_pairwise_distance_all_sessions(
    session_list,
    flexilims_session,
    rf_file=None,
):
    sessions_df_all = pd.DataFrame()
    for session_name in session_list:
        session_df = calculate_pairwise_distance_per_session(
            flexilims_session=flexilims_session,
            session_name=session_name,
            rf_file=rf_file,
        )
        sessions_df_all = pd.concat([sessions_df_all, session_df], axis=0)
        print(f"Finished {session_name}")
    return sessions_df_all


def load_pairwise_distances(
    flexilims_session, cache_path, recompute=False, rf_file=None
):
    """Pairwise distance dataframe of all V1 sessions, `roi_distance` in um.

    With `recompute`, rebuild it from the per-session neurons_df (RF columns from
    `rf_file` if given, see `pipeline_utils.load_neurons_df`) and overwrite the
    pickle at `cache_path`; otherwise read that pickle.
    """
    cache_path = Path(cache_path)
    if recompute:
        from cottage_analysis.summary_analysis import get_session_list

        session_df_all = calculate_pairwise_distance_all_sessions(
            session_list=get_session_list.get_sessions(
                flexilims_session=flexilims_session,
                exclude_openloop=False,
                exclude_pure_closedloop=False,
                v1_only=True,
            ),
            flexilims_session=flexilims_session,
            rf_file=rf_file,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        session_df_all.to_pickle(cache_path)
    else:
        session_df_all = pd.read_pickle(cache_path)
    session_df_all["roi_distance"] = session_df_all["roi_distance"] * PIXEL_SIZE
    return session_df_all


def count_neurons(session_df_all):
    """Number of selected neurons per session, recovered from its pair count.

    A session with k selected neurons has k(k-1)/2 pairs. Sessions with fewer
    than two neurons have no pairs, are absent from the dataframe, and are not
    counted. Counted before any `min_distance` cut, which drops pairs, not neurons.
    """
    n_pairs = session_df_all.dropna(subset=["roi_distance"]).groupby("session").size()
    n_neurons = (1 + np.sqrt(1 + 8 * n_pairs)) / 2
    return n_neurons.round().astype(int)


def hierarchical_binned_means(
    session_df_all, bins=BINS, cols=CI_COLS, n_boot=1000, seed=0, min_distance=10
):
    """Per-bin mean pairwise distance with hierarchically bootstrapped 95% CIs.

    Resampling a neuron PAIR per draw would treat ~1.4e5 pairs per bin as
    independent, when a session with k selected neurons contributes ~k^2/2 of
    them. Here the resampling units are the recordings: mice with replacement,
    then sessions within each drawn mouse, so whole sessions move in and out
    together and the within-session dependence between pairs is preserved.
    Since the statistic is a mean, each session's per-bin sum and count are
    precomputed once and a bootstrap iteration is just a sum over the drawn
    sessions.

    Not resampled: neurons within a session. Doing that properly would mean
    re-deriving the pairs, since pairs cannot be resampled to emulate
    resampling neurons. The between-mouse level dominates here, so omitting it
    errs slightly narrow.

    Pairs closer than `min_distance` um are dropped. The mouse is the session
    name prefix before the first "_".

    Returns:
        bin_centers (n_bins,), means (n_bins, n_cols), ci_low and ci_high
        (n_cols, n_bins), and a dict of counts (sessions, mice, pairs,
        median pairs per bin).
    """
    n_bins = len(bins) - 1
    bin_centers = (bins[1:] + bins[:-1]) / 2

    # Per-(session, bin) sums and counts, in one pass over the pairs
    select_pairs = session_df_all["roi_distance"] > min_distance
    pair_df = session_df_all.loc[
        select_pairs, ["session", "roi_distance"] + [f"{c}_distance" for c in cols]
    ].copy()
    pair_df["bin"] = pd.cut(pair_df["roi_distance"], bins=bins, labels=False)
    pair_df = pair_df.dropna(subset=["bin"])
    pair_df["bin"] = pair_df["bin"].astype(int)

    sessions = np.sort(pair_df["session"].unique())
    grid = dict(index=sessions, columns=range(n_bins))
    grouped = pair_df.groupby(["session", "bin"])
    bin_counts = grouped.size().unstack("bin").reindex(**grid).fillna(0).to_numpy()
    bin_sums = np.stack(
        [
            grouped[f"{c}_distance"]
            .sum()
            .unstack("bin")
            .reindex(**grid)
            .fillna(0)
            .to_numpy()
            for c in cols
        ],
        axis=-1,
    )

    mouse_of = pd.Series(sessions).str.split("_").str[0].to_numpy()
    mice = np.unique(mouse_of)
    sessions_of_mouse = {m: np.flatnonzero(mouse_of == m) for m in mice}

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, n_bins, len(cols)), np.nan)
    for iboot in range(n_boot):
        drawn = np.concatenate(
            [
                rng.choice(
                    sessions_of_mouse[m], size=sessions_of_mouse[m].size, replace=True
                )
                for m in rng.choice(mice, size=mice.size, replace=True)
            ]
        )
        n_drawn = bin_counts[drawn].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[iboot] = bin_sums[drawn].sum(axis=0) / n_drawn[:, None]

    ci_low = np.nanpercentile(boot, 2.5, axis=0).T
    ci_high = np.nanpercentile(boot, 97.5, axis=0).T
    with np.errstate(invalid="ignore", divide="ignore"):
        means = bin_sums.sum(axis=0) / bin_counts.sum(axis=0)[:, None]

    counts = dict(
        n_sessions=len(sessions),
        n_mice=len(mice),
        n_pairs=len(pair_df),
        median_pairs_per_bin=int(np.median(bin_counts.sum(axis=0))),
    )
    return bin_centers, means, ci_low, ci_high, counts


def shuffle_null(session_df_all, bins=BINS, cols=CI_COLS, min_distance=10):
    """Expected per-bin mean if cells were shuffled across positions within session.

    Permuting which cell sits where leaves the set of pairwise values in a
    session unchanged but assigns them to cortical distances at random, so
    the expected mean in any bin is that session's mean over all its pairs.
    Each bin's null is then the average of the session means, weighted by
    how many pairs each session contributes to that bin. That weighting is
    why the null is not flat: sessions with a wider spread of preferences can
    dominate some distance ranges.

    Returns:
        null (n_bins, n_cols)
    """
    n_bins = len(bins) - 1
    value_cols = [f"{c}_distance" for c in cols]
    pair_df = session_df_all.loc[
        session_df_all["roi_distance"] > min_distance,
        ["session", "roi_distance"] + value_cols,
    ].copy()
    session_means = pair_df.groupby("session")[value_cols].mean()
    pair_df["bin"] = pd.cut(pair_df["roi_distance"], bins=bins, labels=False)
    bin_counts = (
        pair_df.dropna(subset=["bin"])
        .groupby(["bin", "session"])
        .size()
        .unstack("session", fill_value=0)
        .reindex(index=range(n_bins), fill_value=0)
    )
    weights = bin_counts.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (
            weights
            @ session_means.loc[bin_counts.columns].to_numpy()
            / weights.sum(axis=1, keepdims=True)
        )


def plot_rf_distance_panels(
    fig,
    rects,
    bin_centers,
    means,
    ci_low,
    ci_high,
    fontsize_dict,
    ylabels=YLABELS,
    null=None,
):
    """One panel per column: mean pairwise distance vs distance between cells.

    `rects` are figure-fraction axes rects, one per column of `means`. `null`
    (n_bins, n_cols), e.g. from `shuffle_null`, is drawn as a dashed line.
    """
    axes = []
    for icol, (rect, ylabel) in enumerate(zip(rects, ylabels)):
        ax = fig.add_axes(rect)
        if null is not None:
            ax.plot(
                bin_centers, null[:, icol], color="gray", linewidth=0.8, linestyle="--"
            )
        ax.plot(bin_centers, means[:, icol], color="k", linewidth=1)
        ax.fill_between(
            bin_centers,
            ci_low[icol],
            ci_high[icol],
            color="k",
            alpha=0.3,
            edgecolor="none",
        )
        ax.set_xticks(np.linspace(0, 860, 3))
        ax.set_xticklabels(
            np.linspace(0, 860, 3).astype("int"), fontsize=fontsize_dict["tick"]
        )
        ax.set_xlim(0, 860)
        # Scale to the bands so the intervals are not clipped
        if icol < 2:
            ymax = np.ceil(np.nanmax(ci_high[icol]))
        else:
            ymax = common_utils.ceil(np.nanmax(ci_high[icol]), 1)
        ax.set_ylim(0, ymax)
        ax.set_yticks([0, ymax])
        ax.tick_params(axis="x", rotation=0, labelsize=fontsize_dict["tick"])
        ax.tick_params(axis="y", labelsize=fontsize_dict["tick"])
        ax.set_xlabel("Distance between cells (μm)", fontsize=fontsize_dict["label"])
        ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
        sns.despine(ax=ax)
        axes.append(ax)
    return axes
