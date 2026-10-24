"""Drifting-grating (SF/TF) helpers for the reviewer grating figure (`figures/fig_reviewer_gratings.ipynb`).

- depth-protocol trials and the panel B example (`load_depth_trials`, `load_depth_example`);
- frame-level SF / TF tuning per running-speed bin (`speed_binned_responses`, `tf_curves_by_speed`,
  `population_tf_curves`, `sftf_model_kept_pairs`);
- held-out population curves per speed bin and for still frames (`build_speed_curves`);
- plotting (`plot_sftf_3x3_grid`, `plot_polar_direction_hist`, `plot_heldout_speed_curves_panel`);
- `bootstrap_spearman`: hierarchical bootstrap of Spearman correlations over sessions.

Exploratory plots (example neurons, population SF/TF maps by preferred depth) are in
`revisions/sftf/sftf_plots.py`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cottage_analysis.analysis import spheres
from cottage_analysis.pipelines import pipeline_utils

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


def speed_binned_responses(
    X,
    response,
    log_rs_median,
    n_bins=3,
    edges_cm_s=None,
    recording=None,
    single_bin_label=None,
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
        single_bin_label (str, optional): all frames in one bin with this label, ignoring the
            running speed (e.g. "Static" for the still frames, whose X[3] is 0).

    Returns:
        list of (label, DataFrame): one table per bin with `SpatialFrequency`,
            `TemporalFrequency`, `Angle` (deg), `trial` ID and the response in column 0.
    """
    log_rs = X[3] + log_rs_median
    if single_bin_label is not None:
        n_bins = 1
        bins = np.zeros(X.shape[1], int)
    elif edges_cm_s is None:
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
    if single_bin_label is not None:
        return [(single_bin_label, frames)]
    cm_s = np.exp(edges) * 100
    return [
        (
            f"{cm_s[i]:.1f}-{cm_s[i + 1]:.1f} cm/s",
            frames[bins == i].reset_index(drop=True),
        )
        for i in range(n_bins)
    ]


FREQ_COLS = {"tf": "TemporalFrequency", "sf": "SpatialFrequency"}


def drift_direction(angle_deg):
    """Drift direction (deg) of a grating of `Angle` `angle_deg`.

    `Angle` is the grating's wave vector and the bars drift towards `Angle` + 180° (stimulus workflow
    `vis-stim-2d/run_SFTF.bonsai` and BonVision, see `revisions/sftf/README.md`). Drift direction in the
    RF-map frame: 0° towards higher azimuth (front to back, the optic flow of forward running), 90° up.
    """
    return (np.asarray(angle_deg) + 180) % 360


def example_condition_means(cache_dir, session, roi):
    """Mean trial response of one ROI to each direction x SF x TF (panel C).

    Returns:
        means (n_angles, n_sf, n_tf), angles (deg, grating `Angle`), sf (cpd), tf (Hz).
    """
    data = pd.read_pickle(cache_dir / f"{session}_sftf_trials.pickle")
    trials = data["trials"].assign(response=data["dff"][:, roi])
    means = trials.groupby(
        ["Angle", "SpatialFrequency", "TemporalFrequency"]
    ).response.mean()
    angles, sf, tf = (np.sort(trials[c].unique()) for c in means.index.names)
    full = pd.MultiIndex.from_product([angles, sf, tf], names=means.index.names)
    return (
        means.reindex(full).to_numpy().reshape(len(angles), len(sf), len(tf)),
        angles,
        sf,
        tf,
    )


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


# ==============================================================================
# Held-out population curves per running-speed bin (panels L-O)
# ==============================================================================

# running-speed bins (cm/s); the first starts at 5 cm/s, the still threshold
SPEED_EDGES_CM_S = [5, 11.31, 22.63, 45.25, 90.51]
TF_SELECT_THRESHOLD = 0.5
P_FREQ_TUNED = 0.05
MIN_CELLS_PER_POINT = 10
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


def bootstrap_spearman(x, y, sessions, n_boots=20000, seed=0):
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


def load_depth_example(cache_dir, session, roi, rebuild=False):
    """Closed-loop depth trials and neurons_df row of one example ROI (panel B), cached in `cache_dir`.

    Only the example ROI is kept (as column 0 of `dff_stim` and the single row of `neurons_df`), so the
    cache stays small.

    Returns:
        dict: `neurons_df` (one row), `trials_depth` (`depth`, `closed_loop`, `RS_stim`, `dff_stim`) and
            `roi` (0, the ROI index in these tables).
    """
    import flexiznam as flz

    path = cache_dir / f"{session}_roi{roi}_depth_example.pickle"
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    flm = flz.get_flexilims_session(PROJECT)
    trials_depth = load_depth_trials(flm, session)
    folder = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=flm, project=PROJECT, conflicts="skip"
    ).path_full.parent
    neurons_df = pd.read_pickle(folder / "neurons_df.pickle")
    td_slim = trials_depth[["depth", "closed_loop", "RS_stim"]].copy()
    td_slim["dff_stim"] = [arr[:, [roi]] for arr in trials_depth["dff_stim"]]
    example = dict(
        neurons_df=neurons_df.iloc[[roi]].reset_index(drop=True),
        trials_depth=td_slim,
        roi=0,
    )
    pd.to_pickle(example, path)
    return example


def build_speed_curves(sftf_root, rebuild=False):
    """Held-out TF and SF tuning curves per running-speed bin and for still frames (panels L-O).

    Cells: significant on the frame-level `base` fit of `precompute_data/sftf_running_models.py`
    (held-out R² above the empirical-null threshold), and for each variable with a significant effect
    of TF (SF) at the kept SF (TF) / direction pairs (`_freq_modulation_p`, trial responses). Kept pairs
    come from the trial-level SF/TF fit (`grating_tuning_cells.csv`, `sftf_model_kept_pairs`).

    For each half of the recordings (`_frame_halves`), the curves of the running frames of one half give
    each cell's peak (of the mean over speed bins) and its value; the curves of the other half, per speed
    bin (`SPEED_EDGES_CM_S`) and for the still frames (bin -1, "Static"), are aligned on that peak and
    divided by that value. The two folds are averaged per cell.

    Args:
        sftf_root (Path): `fig_reviewer_gratings` folder (`grating_tuning_cells.csv`, `cache/`,
            `running_models/` with `frames/`, `frames_still/` and the merged fits).
        rebuild (bool): recompute instead of reading the cache.

    Returns:
        dict: `heldout_means` (mean, SEM, count and bootstrap CI across cells per variable, depth
            group, speed bin and frequency relative to the peak; points with fewer than
            `MIN_CELLS_PER_POINT` cells dropped), `heldout_counts` ((cells, sessions, mice) per variable
            and depth group), `mean_peaks` (mean peak position per speed bin) and `speed_labels`.
    """
    from cottage_analysis.analysis import common_utils

    cache_dir = sftf_root / "cache"
    frames_dir = sftf_root / "running_models" / "frames"
    still_dir = sftf_root / "running_models" / "frames_still"
    cache = cache_dir / f"speed_curves_v{SPEED_CURVES_VERSION}.pickle"
    if cache.exists() and not rebuild:
        return pd.read_pickle(cache)

    cells = pd.read_csv(sftf_root / "grating_tuning_cells.csv")
    pickle_files = sorted(
        (sftf_root / "running_models").glob("*_sftf_running_models.pickle")
    )
    rs_fits = pd.concat(
        [pd.read_pickle(f) for f in pickle_files if not f.name.startswith(".")],
        ignore_index=True,
    )
    fit_cols = [c for c in cells.columns if c.startswith("fit_")]
    rs_fits = rs_fits.merge(
        cells[["session", "roi", "depth_tuned"] + fit_cols],
        on=["session", "roi"],
        how="left",
        validate="one_to_one",
    )
    common_utils.add_rsq_significance(rs_fits, {"base": "base_test_rsq"}, verbose=False)
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
                sc, sf_v, tf_v, ang_v, variable=var, threshold=TF_SELECT_THRESHOLD
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
                        p=_freq_modulation_p(
                            t_sess["trials"], t_sess["dff"][:, sc.roi.to_numpy()], var
                        ),
                    )
                )
            )
    p_mod = pd.concat(p_mod, ignore_index=True)
    freq_tuned = p_mod[p_mod.p < P_FREQ_TUNED][["variable", "session", "roi"]]

    keys = ["variable", "depth_tuned", "speed_bin", "speed", "delta"]
    heldout_pop = (
        pd.concat(heldout_rows, ignore_index=True)
        .merge(freq_tuned, on=["variable", "session", "roi"])
        .groupby(
            ["variable", "depth_tuned", "session", "roi", "speed_bin", "speed", "delta"]
        )
        .resp.mean()
        .reset_index()
    )

    def _boot_ci(values):
        lo, hi = common_utils.get_bootstrap_ci(values.dropna().to_numpy())
        return pd.Series(dict(ci_low=lo[0], ci_high=hi[0]))

    heldout_means = (
        heldout_pop.groupby(keys).resp.agg(["mean", "sem", "count"]).reset_index()
    )
    heldout_means = heldout_means[heldout_means["count"] >= MIN_CELLS_PER_POINT]
    heldout_means = heldout_means.merge(
        heldout_pop.groupby(keys).resp.apply(_boot_ci).unstack().reset_index(), on=keys
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

    curves = dict(
        heldout_means=heldout_means,
        heldout_counts=heldout_counts,
        mean_peaks=mean_peaks,
        speed_labels=speed_labels,
    )
    pd.to_pickle(curves, cache)
    return curves


def save_with_vector_image(fig, image, vector_pdf, pdf_path, png_path=None, dpi=300):
    """Save `fig` as PDF with a raster image replaced by its vector version, and as PNG.

    Matplotlib cannot place vector graphics in an axes: `image` (an `AxesImage` of a raster copy of
    `vector_pdf`) is drawn in the PNG, hidden in the PDF, and the first page of `vector_pdf` is merged
    onto the PDF at the same position and size (`pypdf`). Both files are cropped to the same tight
    bounding box, as `bbox_inches="tight"`.
    """
    from matplotlib.transforms import Bbox
    from pypdf import PdfReader, PdfWriter, Transformation

    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer).padded(plt.rcParams["savefig.pad_inches"])
    extent = Bbox(image.get_window_extent(renderer).get_points() / fig.dpi)  # inches
    if png_path is not None:
        fig.savefig(png_path, dpi=dpi, bbox_inches=bbox)
    image.set_visible(False)
    fig.savefig(pdf_path, bbox_inches=bbox)
    image.set_visible(True)

    page = PdfReader(pdf_path).pages[0]
    insert = PdfReader(vector_pdf).pages[0]
    scale_x = extent.width * 72 / float(insert.mediabox.width)
    scale_y = extent.height * 72 / float(insert.mediabox.height)
    page.merge_transformed_page(
        insert,
        Transformation()
        .translate(-float(insert.mediabox.left), -float(insert.mediabox.bottom))
        .scale(scale_x, scale_y)
        .translate((extent.x0 - bbox.x0) * 72, (extent.y0 - bbox.y0) * 72),
    )
    writer = PdfWriter()
    writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)


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
