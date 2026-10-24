"""One row per `iscell` ROI of the six drifting-grating sessions: grating tuning, depth and RS/OF tuning.

Input of `figures/fig_reviewer_gratings.ipynb` (and of `revisions/sftf/`). Writes, under the figure folder
`<processed>/v1_manuscript_figures/ver_rev1/fig_reviewer_gratings/`:

- `cache/<session>_sftf_trials.pickle`: trial table of the SFTF recordings and the (n_trials, n_rois) mean
  dF/F of each trial over `[onset + SKIP, next_onset)` volumes. Trials are split where the parameter-log time
  (`stimulus_harptime`) changes; the last trial of a recording gets the median trial length.
- `cache/<session>_neurons_slim.pickle`: the columns of `load_session_neurons` used here.
- `grating_tuning_cells.csv`:
  - model-free tuning at each ROI's preferred SF/TF: vector-sum gOSI / gDSI of the rectified direction curve,
    preferred orientation / direction, their shuffle p-values (direction labels permuted among the trials of
    that SF/TF) and the one-way ANOVA p across the 288 conditions;
  - depth selectivity and preferred depth (closed-loop depth tuning, one-sided Spearman p < 0.05), the RS/OF
    Gaussian fit (preferred optic flow in deg/s and running speed in m/s, held-out R²);
  - the trial-level SF/TF fit (`neurons_df_sftf.pickle`, columns prefixed `fit_`), preferred grating speed
    (TF0 / SF0, deg/s) and the selections: `sftf_tuned` (held-out Spearman, one-sided p < 0.05), `rsof_tuned`
    (RS/OF held-out R² above the empirical-null threshold, as figure 2), `responsive`, `ori_selective`,
    `dir_selective`.

Moved from the analysis cells of `revisions/sftf/sftf_exploratory_analysis.ipynb`; same output.

Usage:
    python sftf_grating_cells.py [--rebuild]
"""

import argparse

import flexiznam as flz
import numpy as np
import pandas as pd
from scipy import stats

from cottage_analysis.analysis import common_utils
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.preprocessing import synchronisation
from v1_depth_map.figure_utils.rf_distance import load_session_neurons
from v1_depth_map.paths import get_figures_roots

PROJECT = "hey2_3d-vision_foodres_20220101"
SESSIONS = [
    "PZAH10.2d_S20230920",
    "PZAH10.2d_S20230922",
    "PZAH10.2f_S20230817",
    "PZAH10.2f_S20230912",
    "PZAH10.2f_S20230914",
    "PZAH10.2f_S20230924",
]
PHOTODIODE_PROTOCOL = 5  # all PZAH mice
TRACES = {"anatomical_only": 3, "ast_neuropil": False}
PARAMS = ["SpatialFrequency", "TemporalFrequency", "Angle"]
SKIP = 8  # onset volumes skipped (0.53 s at 15.2 Hz); the window ends at the next onset
MIN_VOLUMES = 10  # drop trials shorter than this (truncated at the end of a recording)
P_RESPONSIVE = 0.01  # ANOVA across conditions
P_SELECTIVE = 0.05  # gOSI / gDSI shuffle
N_SHUFFLES = 1000
RSOF_SIG_PERCENTILE = (
    95  # empirical-null threshold of the RS/OF held-out R², as figure 2
)
NEURONS_COLUMNS = [
    "roi",
    "iscell",
    "depth_tuned",
    "preferred_depth_closedloop",
    "preferred_OF_closedloop_g2d",
    "preferred_RS_closedloop_g2d",
    "rsof_test_rsq_closedloop_g2d",
    "rsof_popt_closedloop_g2d",
]


def recording_trials(flm, recording):
    """Trial table and (n_trials, n_rois) mean dF/F of one SFTF recording."""
    vs_df = synchronisation.generate_vs_df(
        recording=recording,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        flexilims_session=flm,
        project=PROJECT,
    )
    img_df = synchronisation.generate_imaging_df(
        vs_df=vs_df,
        recording=recording,
        flexilims_session=flm,
        filter_datasets=TRACES,
        return_volumes=True,
    ).reset_index(drop=True)
    dff = np.vstack(img_df.dffs.values)  # (n_volumes, n_rois)

    stim = img_df["stimulus_harptime"]
    onset = (stim.notna() & (stim != stim.shift())).values
    starts = np.flatnonzero(onset)
    ends = np.append(starts[1:], starts[-1] + int(np.median(np.diff(starts))))
    starts, ends = starts + SKIP, np.minimum(ends, len(img_df))
    keep = ends - starts >= MIN_VOLUMES

    trials = img_df.loc[onset, PARAMS].reset_index(drop=True)
    trials["n_volumes"] = ends - starts
    trials["recording"] = recording["name"]
    resp = np.stack([dff[s:e].mean(axis=0) for s, e in zip(starts, ends)])
    return trials[keep].reset_index(drop=True), resp[keep]


def session_trials(flm, session, cache, rebuild=False):
    """All SFTF trials of a session, cached in `cache`."""
    path = cache / f"{session}_sftf_trials.pickle"
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )  # indexed by name, which the sync functions read as `recording.name`
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()
    out = [recording_trials(flm, rec) for _, rec in recordings.iterrows()]
    data = dict(
        trials=pd.concat([o[0] for o in out], ignore_index=True),
        dff=np.concatenate([o[1] for o in out], axis=0),
    )
    pd.to_pickle(data, path)
    return data


def vector_metrics(curves, theta):
    """gOSI, gDSI, preferred orientation and direction (deg) of rectified curves (..., n_dir)."""
    r = np.clip(curves, 0, None)
    total = r.sum(axis=-1)
    v1 = (r * np.exp(1j * theta)).sum(axis=-1)
    v2 = (r * np.exp(2j * theta)).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gosi, gdsi = np.abs(v2) / total, np.abs(v1) / total
    pref_ori = np.rad2deg(np.angle(v2)) / 2 % 180
    pref_dir = np.rad2deg(np.angle(v1)) % 360
    return gosi, gdsi, pref_ori, pref_dir


def session_tuning(trials, resp, rng):
    """Model-free tuning of each ROI at its preferred SF/TF, shuffle and ANOVA p-values."""
    values = [np.unique(trials[p]) for p in PARAMS]
    n_sf, n_tf, n_dir = (len(v) for v in values)
    i_sf, i_tf, i_dir = (np.searchsorted(v, trials[p]) for v, p in zip(values, PARAMS))
    i_sftf = i_sf * n_tf + i_tf
    cond = i_sftf * n_dir + i_dir
    n_cond, n_roi = n_sf * n_tf * n_dir, resp.shape[1]
    theta = np.deg2rad(values[2])

    counts = np.bincount(cond, minlength=n_cond)
    sums = np.zeros((n_cond, n_roi))
    np.add.at(sums, cond, resp)
    with np.errstate(invalid="ignore"):
        means = (sums / counts[:, None]).reshape(n_sf * n_tf, n_dir, n_roi)
    best = np.nanargmax(np.nanmean(means, axis=1), axis=0)  # preferred SF/TF per ROI
    curves = means[best, :, np.arange(n_roi)]  # (n_roi, n_dir)
    gosi, gdsi, pref_ori, pref_dir = vector_metrics(curves, theta)

    # shuffle direction labels among the trials of each ROI's preferred SF/TF
    p_osi, p_dsi = np.full(n_roi, np.nan), np.full(n_roi, np.nan)
    for k in np.unique(best):
        rois = np.flatnonzero(best == k)
        tr = np.flatnonzero(i_sftf == k)
        labels = np.tile(i_dir[tr], (N_SHUFFLES, 1))
        onehot = np.eye(n_dir)[
            rng.permuted(labels, axis=1)
        ]  # (n_shuffles, n_trials, n_dir)
        onehot /= np.bincount(i_dir[tr], minlength=n_dir)
        shuffled = np.einsum("std,tr->srd", onehot, resp[np.ix_(tr, rois)])
        osi_sh, dsi_sh, _, _ = vector_metrics(shuffled, theta)
        p_osi[rois] = (1 + (osi_sh >= gosi[rois]).sum(axis=0)) / (N_SHUFFLES + 1)
        p_dsi[rois] = (1 + (dsi_sh >= gdsi[rois]).sum(axis=0)) / (N_SHUFFLES + 1)

    p_resp = stats.f_oneway(*[resp[cond == c] for c in range(n_cond)], axis=0).pvalue
    return pd.DataFrame(
        dict(
            roi=np.arange(n_roi),
            pref_sf=values[0][best // n_tf],
            pref_tf=values[1][best % n_tf],
            gosi=gosi,
            gdsi=gdsi,
            pref_ori=pref_ori,
            pref_dir=pref_dir,
            p_osi=p_osi,
            p_dsi=p_dsi,
            p_resp=p_resp,
        )
    )


def session_neurons(flm, session, cache, rebuild=False):
    """Columns of `load_session_neurons` used here, cached (the full neurons_df is ~150-325 MB)."""
    path = cache / f"{session}_neurons_slim.pickle"
    if path.exists() and not rebuild:
        return pd.read_pickle(path)
    neurons_df, _ = load_session_neurons(flm, session)
    slim = neurons_df[NEURONS_COLUMNS].reset_index(drop=True)
    slim.to_pickle(path)
    return slim


def load_sftf_fit(flm, session):
    """Trial-level SF/TF fit of a session (`neurons_df_sftf.pickle`, cross-validated)."""
    folder = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=flm, project=PROJECT, conflicts="skip"
    ).path_full.parent
    fit = pd.read_pickle(folder / "neurons_df_sftf.pickle")
    assert (
        "sftf_test_spearmanr_pval" in fit
    ), f"{session}: neurons_df_sftf has no cross-validation"
    fit.insert(0, "roi", np.arange(len(fit)))
    fit.insert(0, "session", session)
    return fit.drop(columns=["sftf_test_popts"])


def main(rebuild=False):
    flm = flz.get_flexilims_session(PROJECT)
    _, save_root = get_figures_roots(flm, fig_subdir="fig_reviewer_gratings")
    cache = save_root / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    cells = []
    for session in SESSIONS:
        data = session_trials(flm, session, cache, rebuild=rebuild)
        neurons_df = session_neurons(flm, session, cache, rebuild=rebuild)
        tuning = session_tuning(data["trials"], data["dff"], rng)
        assert len(tuning) == len(
            neurons_df
        ), f"{session}: ROI count differs from neurons_df"
        tuning["session"] = session
        tuning["iscell"] = neurons_df.iscell.values.astype(bool)
        tuning["depth_tuned"] = neurons_df.depth_tuned.values.astype(bool)
        tuning["preferred_depth"] = neurons_df.preferred_depth_closedloop.values
        # RS/OF Gaussian fit of the depth protocol, rad/s -> deg/s
        tuning["preferred_of"] = np.degrees(
            neurons_df.preferred_OF_closedloop_g2d.astype(float).values
        )
        tuning["rsof_test_rsq"] = neurons_df.rsof_test_rsq_closedloop_g2d.astype(
            float
        ).values
        tuning["preferred_rs"] = neurons_df.preferred_RS_closedloop_g2d.astype(
            float
        ).values  # m/s
        cells.append(tuning)
    cells = pd.concat(cells, ignore_index=True)
    cells = cells[cells.iscell].reset_index(drop=True)
    cells["group"] = np.where(cells.depth_tuned, "depth tuned", "not depth tuned")
    cells["responsive"] = cells.p_resp < P_RESPONSIVE
    cells["ori_selective"] = cells.responsive & (cells.p_osi < P_SELECTIVE)
    cells["dir_selective"] = cells.responsive & (cells.p_dsi < P_SELECTIVE)

    fits = pd.concat([load_sftf_fit(flm, s) for s in SESSIONS], ignore_index=True)
    cells = cells.merge(
        fits.rename(
            columns={c: f"fit_{c}" for c in fits.columns if c not in ("session", "roi")}
        ),
        on=["session", "roi"],
        how="left",
        validate="one_to_one",
    )
    cells["pref_speed"] = np.exp(cells.fit_tf0 - cells.fit_sf0)  # deg/s
    cells["fit_dir"] = np.rad2deg(cells.fit_alpha0) % 360
    cells["fit_ori"] = cells.fit_dir % 180
    common_utils.add_one_sided_spearman_significance(
        cells,
        rval_col="fit_sftf_test_spearmanr_rval",
        pval_col="fit_sftf_test_spearmanr_pval",
        out_col="sftf_tuned",
    )
    rsof_thresholds = common_utils.add_rsq_significance(
        cells, {"g2d": "rsof_test_rsq"}, percentile=RSOF_SIG_PERCENTILE, verbose=False
    )
    cells["rsof_tuned"] = cells["rsof_test_rsq_sig"]
    cells.to_csv(save_root / "grating_tuning_cells.csv", index=False)

    print(f"{len(cells)} iscell ROIs -> {save_root / 'grating_tuning_cells.csv'}")
    print(f"depth selective: {cells.depth_tuned.sum()}")
    print(f"SF/TF tuned (cross-validated): {cells.sftf_tuned.sum()}")
    print(
        f"RS/OF tuned (test R² > empirical-null threshold {rsof_thresholds['g2d']:.4f}): "
        f"{cells.rsof_tuned.sum()}"
    )
    print(
        f"SF/TF tuned and depth selective: {(cells.sftf_tuned & cells.depth_tuned).sum()}"
    )
    print(f"SF/TF tuned and RS/OF tuned: {(cells.sftf_tuned & cells.rsof_tuned).sum()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="re-sync the recordings and reload neurons_df instead of reading the caches",
    )
    main(rebuild=p.parse_args().rebuild)
