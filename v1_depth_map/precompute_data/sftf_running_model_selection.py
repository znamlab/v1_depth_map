"""Running-speed model selection on 10 high-SNR V1 cells.

Compares 9 models of grating responses frame by frame:
    base:               offset + A * G(sf0, tf0) * D(alpha0)
    gain:               base with amplitude A * exp(g * r)
    offset:             base with baseline offset + b * r
    gain_offset:        both multiplicative gain and additive offset
    tf:                 base with log TF0 = tf0 + a_tf * r (no gain/offset)
    gain_offset_tf:     gain + offset + log TF0 = tf0 + a_tf * r
    gain_offset_sf:     gain + offset + log SF0 = sf0 + a_sf * r
    gain_offset_tf_sf:  gain + offset + both TF and SF shifts
    gain_offset_speed:  gain + offset + shift along speed axis (tf0 + a*r, sf0 - a*r)

where r = log(RS) - median_log_rs.

Samples are running frames (RS > 1 cm/s), dF/F shifted by LAG frames (default 5).
Each model is fitted with nested warm starts and evaluated using leave-one-recording-out
cross-validation (5 folds).

Usage:
    python sftf_running_model_selection.py prepare --out-dir DIR
    python sftf_running_model_selection.py fit [SESSION ROI] --out-dir DIR [--n-jobs 6]
    python sftf_running_model_selection.py summary --out-dir DIR
    python sftf_running_model_selection.py simulate --out-dir DIR [--n-jobs 6]
    python sftf_running_model_selection.py robustness --out-dir DIR [--n-jobs 6]
"""

import argparse
import time
from functools import partial
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from cottage_analysis.analysis import common_utils
from cottage_analysis.analysis.fit_gaussian_blob import GratingParams, grating_tuning
from cottage_analysis.preprocessing import synchronisation

PROJECT = "hey2_3d-vision_foodres_20220101"
PHOTODIODE_PROTOCOL = 5
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}
PARAMS = ["SpatialFrequency", "TemporalFrequency", "Angle"]
RS_THR = 0.01  # m/s
LAG = 5
MIN_SIGMA = 0.25
MAX_LOG_KAPPA = 6.0  # avoids exp overflow in direction_tuning (kappa <= 403)
MIN_LOG_SIGMA2 = -20.0  # bounds flat valley where exp(log_sigma2) << min_sigma
SLOPE_BOUND = 5.0
MAXFEV = 5000

CHOSEN_CELLS = [
    ("PZAH10.2d_S20230920", 220, "Top cell in session, mouse PZAH10.2d", False),
    ("PZAH10.2d_S20230922", 171, "Top cell in session, mouse PZAH10.2d", False),
    (
        "PZAH10.2f_S20230817",
        62,
        "Top cell in session, depth-tuned, mouse PZAH10.2f",
        True,
    ),
    (
        "PZAH10.2f_S20230817",
        259,
        "Rank 2 in session, depth-tuned, mouse PZAH10.2f",
        True,
    ),
    (
        "PZAH10.2f_S20230817",
        214,
        "Rank 4 in session, not depth-tuned, mouse PZAH10.2f",
        False,
    ),
    (
        "PZAH10.2f_S20230817",
        544,
        "Thesis example neuron, not depth-tuned, mouse PZAH10.2f",
        False,
    ),
    ("PZAH10.2f_S20230912", 391, "Top cell in session, mouse PZAH10.2f", False),
    (
        "PZAH10.2f_S20230914",
        8,
        "Top cell in session, depth-tuned, mouse PZAH10.2f",
        True,
    ),
    ("PZAH10.2f_S20230924", 206, "Top cell in session, mouse PZAH10.2f", False),
    ("PZAH10.2f_S20230924", 269, "Rank 2 in session, mouse PZAH10.2f", False),
]

MODELS = {
    "base": (),
    "gain": ("g",),
    "offset": ("b",),
    "gain_offset": ("g", "b"),
    "tf": ("a_tf",),
    "gain_offset_tf": ("g", "b", "a_tf"),
    "gain_offset_sf": ("g", "b", "a_sf"),
    "gain_offset_tf_sf": ("g", "b", "a_tf", "a_sf"),
    "gain_offset_speed": ("g", "b", "a"),
}


def eval_model(m_name, X, *p, min_sigma=MIN_SIGMA):
    """Evaluate grating response for candidate model `m_name` given X (4, N) and params."""
    (
        log_amp,
        sf0,
        tf0,
        log_sigma_x2,
        log_sigma_y2,
        theta,
        offset,
        alpha0,
        log_kappa,
        dsi,
    ) = p[:10]
    extra = p[10:]
    r = X[3]

    if m_name == "base":
        a, b, sf, tf = log_amp, offset, sf0, tf0
    elif m_name == "gain":
        a, b, sf, tf = log_amp + extra[0] * r, offset, sf0, tf0
    elif m_name == "offset":
        a, b, sf, tf = log_amp, offset + extra[0] * r, sf0, tf0
    elif m_name == "gain_offset":
        a, b, sf, tf = log_amp + extra[0] * r, offset + extra[1] * r, sf0, tf0
    elif m_name == "tf":
        a, b, sf, tf = log_amp, offset, sf0, tf0 + extra[0] * r
    elif m_name == "gain_offset_tf":
        a, b, sf, tf = (
            log_amp + extra[0] * r,
            offset + extra[1] * r,
            sf0,
            tf0 + extra[2] * r,
        )
    elif m_name == "gain_offset_sf":
        a, b, sf, tf = (
            log_amp + extra[0] * r,
            offset + extra[1] * r,
            sf0 + extra[2] * r,
            tf0,
        )
    elif m_name == "gain_offset_tf_sf":
        a, b, sf, tf = (
            log_amp + extra[0] * r,
            offset + extra[1] * r,
            sf0 + extra[3] * r,
            tf0 + extra[2] * r,
        )
    elif m_name == "gain_offset_speed":
        a, b, sf, tf = (
            log_amp + extra[0] * r,
            offset + extra[1] * r,
            sf0 - extra[2] * r,
            tf0 + extra[2] * r,
        )
    else:
        raise ValueError(f"Unknown model: {m_name}")

    return grating_tuning(
        (X[0], X[1], X[2]),
        a,
        sf,
        tf,
        log_sigma_x2,
        log_sigma_y2,
        theta,
        b,
        alpha0,
        log_kappa,
        dsi,
        min_sigma,
    )


def get_bounds(X, m_name):
    """Lower and upper parameter bounds for model `m_name`."""
    lo = [
        -np.inf,
        X[0].min() - 1,
        X[1].min() - 1,
        MIN_LOG_SIGMA2,
        MIN_LOG_SIGMA2,
        0.0,
        -np.inf,
        0.0,
        -np.inf,
        0.0,
    ]
    hi = [
        np.inf,
        X[0].max() + 1,
        X[1].max() + 1,
        np.inf,
        np.inf,
        np.pi / 2,
        np.inf,
        2 * np.pi,
        MAX_LOG_KAPPA,
        1.0,
    ]
    for param in MODELS[m_name]:
        if param == "b":
            lo.append(-np.inf)
            hi.append(np.inf)
        else:
            lo.append(-SLOPE_BOUND)
            hi.append(SLOPE_BOUND)
    return lo, hi


def random_p0(X, y):
    """Random start of the base model, SF0 / TF0 / direction at best stimulus marginals."""
    y_ser = pd.Series(y)
    return list(
        GratingParams(
            log_amplitude=float(np.random.normal()),
            sf0=float(y_ser.groupby(X[0]).mean().idxmax()),
            tf0=float(y_ser.groupby(X[1]).mean().idxmax()),
            log_sigma_x2=float(np.random.normal()),
            log_sigma_y2=float(np.random.normal()),
            theta=float(np.random.uniform(0, 0.5 * np.pi)),
            offset=float(np.random.normal()),
            alpha0=float(y_ser.groupby(X[2]).mean().idxmax()),
            log_kappa=float(
                np.clip(np.random.normal(1.0, 1.0), -2.0, MAX_LOG_KAPPA - 0.5)
            ),
            dsi=float(np.random.uniform(0, 1)),
        )
    )


def best_fit(m_name, X, y, starts, maxfev=MAXFEV):
    """Find best fit (highest R2) of curve_fit runs from given starts."""
    lo, hi = get_bounds(X, m_name)
    best = (np.full(len(lo), np.nan), -np.inf, False)
    func = partial(eval_model, m_name)
    for p0 in starts:
        p0 = np.clip(p0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        hit_maxfev = False
        try:
            popt, _ = curve_fit(func, X, y, p0=p0, bounds=(lo, hi), maxfev=maxfev)
        except RuntimeError:
            hit_maxfev = True
            continue
        except ValueError:
            continue
        pred = func(X, *popt)
        rsq = common_utils.calculate_r_squared(y, pred)
        if rsq > best[1]:
            best = (popt, rsq, hit_maxfev)
    return best[0], best[1], best[2]


def fit_models(X, y, seed=42):
    """Fit all 9 models on frames X, y with nested warm starts.

    Returns dict of {model_name: (popt, rsq, hit_maxfev)}.
    """
    np.random.seed(seed)
    fits = {}

    # 1. Base (5 random starts)
    starts_base = [random_p0(X, y) for _ in range(5)]
    fits["base"] = best_fit("base", X, y, starts_base)
    base_popt = (
        fits["base"][0] if np.all(np.isfinite(fits["base"][0])) else random_p0(X, y)
    )

    # 2. Gain: warm start from base (g=0) + 3 random starts
    starts_gain = [list(base_popt) + [0.0]] + [
        random_p0(X, y) + [float(np.random.uniform(-1, 1))] for _ in range(3)
    ]
    fits["gain"] = best_fit("gain", X, y, starts_gain)
    gain_popt = (
        fits["gain"][0]
        if np.all(np.isfinite(fits["gain"][0]))
        else list(base_popt) + [0.0]
    )

    # 3. Offset: warm start from base (b=0) + 3 random starts
    starts_offset = [list(base_popt) + [0.0]] + [
        random_p0(X, y) + [float(np.random.uniform(-1, 1))] for _ in range(3)
    ]
    fits["offset"] = best_fit("offset", X, y, starts_offset)
    offset_popt = (
        fits["offset"][0]
        if np.all(np.isfinite(fits["offset"][0]))
        else list(base_popt) + [0.0]
    )

    # 4. Gain-offset: warm starts from gain (b=0) and offset (g=0) + 3 random starts
    starts_go = [
        list(gain_popt[:10]) + [gain_popt[10], 0.0],
        list(offset_popt[:10]) + [0.0, offset_popt[10]],
    ] + [random_p0(X, y) + list(np.random.uniform(-1, 1, 2)) for _ in range(3)]
    fits["gain_offset"] = best_fit("gain_offset", X, y, starts_go)
    go_popt = (
        fits["gain_offset"][0]
        if np.all(np.isfinite(fits["gain_offset"][0]))
        else list(base_popt) + [0.0, 0.0]
    )

    # 5. TF: warm start from base (a_tf=0) + 3 random starts
    starts_tf = [list(base_popt) + [0.0]] + [
        random_p0(X, y) + [float(np.random.uniform(-1, 1))] for _ in range(3)
    ]
    fits["tf"] = best_fit("tf", X, y, starts_tf)

    # 6. Gain-offset-TF: warm start from gain_offset (a_tf=0) + 3 random starts
    starts_gotf = [list(go_popt) + [0.0]] + [
        random_p0(X, y) + list(np.random.uniform(-1, 1, 3)) for _ in range(3)
    ]
    fits["gain_offset_tf"] = best_fit("gain_offset_tf", X, y, starts_gotf)
    gotf_popt = (
        fits["gain_offset_tf"][0]
        if np.all(np.isfinite(fits["gain_offset_tf"][0]))
        else list(go_popt) + [0.0]
    )

    # 7. Gain-offset-SF: warm start from gain_offset (a_sf=0) + 3 random starts
    starts_gosf = [list(go_popt) + [0.0]] + [
        random_p0(X, y) + list(np.random.uniform(-1, 1, 3)) for _ in range(3)
    ]
    fits["gain_offset_sf"] = best_fit("gain_offset_sf", X, y, starts_gosf)
    gosf_popt = (
        fits["gain_offset_sf"][0]
        if np.all(np.isfinite(fits["gain_offset_sf"][0]))
        else list(go_popt) + [0.0]
    )

    # 8. Gain-offset-TF-SF: warm starts from gain_offset_tf (a_sf=0) & gain_offset_sf (a_tf=0) + 3 random
    starts_gotfsf = [
        list(gotf_popt) + [0.0],
        list(gosf_popt[:12]) + [0.0, gosf_popt[12]],
    ] + [random_p0(X, y) + list(np.random.uniform(-1, 1, 4)) for _ in range(3)]
    fits["gain_offset_tf_sf"] = best_fit("gain_offset_tf_sf", X, y, starts_gotfsf)

    # 9. Gain-offset-speed: warm start from gain_offset (a=0) + 3 random starts
    starts_gospeed = [list(go_popt) + [0.0]] + [
        random_p0(X, y) + list(np.random.uniform(-1, 1, 3)) for _ in range(3)
    ]
    fits["gain_offset_speed"] = best_fit("gain_offset_speed", X, y, starts_gospeed)

    return fits


def fit_roi(X, y, recording, n_jobs=1, seed=42):
    """Full-data and leave-one-recording-out 5-fold CV for one ROI across all 9 models."""
    recs = np.unique(recording)
    subsets = [np.ones(len(y), bool)] + [recording != rec for rec in recs]

    if n_jobs == 1:
        fits = [fit_models(X[:, s], y[s], seed=seed + i) for i, s in enumerate(subsets)]
    else:
        from joblib import Parallel, delayed

        fits = Parallel(n_jobs=n_jobs)(
            delayed(fit_models)(X[:, s], y[s], seed=seed + i)
            for i, s in enumerate(subsets)
        )

    full, folds = fits[0], fits[1:]
    preds = {m: np.full(len(y), np.nan) for m in MODELS}

    for rec, fold in zip(recs, folds):
        test = recording == rec
        for m in MODELS:
            if np.all(np.isfinite(fold[m][0])):
                preds[m][test] = eval_model(m, X[:, test], *fold[m][0])

    row = {}
    n_samples = len(y)

    for m in MODELS:
        names = list(GratingParams._fields) + list(MODELS[m])
        popt, train_rsq, maxfev_flag = full[m]
        row.update({f"{m}_{n}": float(v) for n, v in zip(names, popt)})
        row[f"{m}_rsq"] = float(train_rsq)
        row[f"{m}_hit_maxfev"] = bool(maxfev_flag)

        # In-sample residuals and AIC/BIC
        full_pred = (
            eval_model(m, X, *popt) if np.all(np.isfinite(popt)) else np.zeros_like(y)
        )
        rss = float(np.sum((y - full_pred) ** 2))
        k = len(names)
        row[f"{m}_rss"] = rss
        row[f"{m}_k"] = k
        # Standard Gaussian log-likelihood AIC / BIC from RSS
        if rss > 0:
            row[f"{m}_aic"] = float(n_samples * np.log(rss / n_samples) + 2 * k)
            row[f"{m}_bic"] = float(
                n_samples * np.log(rss / n_samples) + k * np.log(n_samples)
            )
        else:
            row[f"{m}_aic"] = np.nan
            row[f"{m}_bic"] = np.nan

        # Held-out cross-validated metrics
        ok = np.isfinite(preds[m])
        if ok.sum() > 0:
            row[f"{m}_test_rsq"] = float(
                common_utils.calculate_r_squared(y[ok], preds[m][ok])
            )
            rval, pval = spearmanr(y[ok], preds[m][ok])
            row[f"{m}_test_spearmanr_rval"] = float(rval)
            row[f"{m}_test_spearmanr_pval"] = float(pval)
        else:
            row[f"{m}_test_rsq"] = np.nan
            row[f"{m}_test_spearmanr_rval"] = np.nan
            row[f"{m}_test_spearmanr_pval"] = np.nan

    return row


def extract_session_frames(flm, session, lag=LAG, rs_thr=RS_THR, smooth_rs_sec=0.0):
    """Extract running frames, dF/F, recording IDs, and log RS for a session."""
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()
    xs, dffs, recs = [], [], []

    for _, rec in recordings.iterrows():
        vs_df = synchronisation.generate_vs_df(
            recording=rec,
            photodiode_protocol=PHOTODIODE_PROTOCOL,
            flexilims_session=flm,
            project=PROJECT,
        )
        img = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=rec,
            flexilims_session=flm,
            filter_datasets=FILTER_TRACES,
            return_volumes=True,
        ).reset_index(drop=True)

        rs_raw = (img.mouse_z_harp.diff() / img.mouse_z_harptime.diff()).to_numpy()
        if smooth_rs_sec > 0:
            # ~15.2 Hz imaging rate: smooth over window
            fps = 15.2
            window = max(1, int(round(smooth_rs_sec * fps)))
            rs = (
                pd.Series(rs_raw)
                .rolling(window, center=True, min_periods=1)
                .mean()
                .to_numpy()
            )
        else:
            rs = rs_raw

        stim = img[PARAMS].to_numpy(float)
        dff = np.vstack(img.dffs.values)

        stim, rs, dff = stim[:-lag], rs[:-lag], dff[lag:]
        keep = np.isfinite(stim).all(axis=1) & np.isfinite(rs) & (rs > rs_thr)
        xs.append(
            np.c_[np.log(stim[keep, :2]), np.deg2rad(stim[keep, 2]), np.log(rs[keep])]
        )
        dffs.append(dff[keep])
        recs.append(np.full(keep.sum(), rec["name"]))

    X = np.concatenate(xs).T
    log_rs_median = float(np.median(X[3]))
    X[3] -= log_rs_median
    return X, np.concatenate(dffs), np.concatenate(recs), log_rs_median


def prepare(out_dir):
    """Extract and cache running frames for the 10 chosen cells across all 6 sessions."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    flm = flz.get_flexilims_session(PROJECT)

    cell_map = {}
    for sess, roi, why, depth_tuned in CHOSEN_CELLS:
        cell_map.setdefault(sess, []).append(roi)

    for sess, rois in cell_map.items():
        target = frames_dir / f"{sess}.npz"
        if target.exists():
            print(f"{target} exists, skipping")
            continue
        print(f"Preparing frames for {sess} (ROIs: {rois})...")
        X, dff, recs, log_rs_med = extract_session_frames(flm, sess)
        dff_rois = dff[:, rois]
        np.savez(
            target,
            X=X,
            dff=dff_rois,
            rois=np.array(rois),
            recording=recs,
            log_rs_median=log_rs_med,
        )
        print(f"Saved {sess}: frames={X.shape[1]}, rois={rois} -> {target}")


def fit_single_roi(session, roi, out_dir, n_jobs=6):
    """Fit all 9 models on one ROI and save result pickle."""
    out_dir = Path(out_dir)
    rois_dir = out_dir / "rois"
    rois_dir.mkdir(parents=True, exist_ok=True)
    target = rois_dir / f"{session}_{roi}.pickle"
    if target.exists():
        print(f"{target} already exists, skipping")
        return pd.read_pickle(target)

    frames_file = out_dir / "frames" / f"{session}.npz"
    d = np.load(frames_file)
    (col,) = np.flatnonzero(d["rois"] == roi)
    X = d["X"]
    y = d["dff"][:, col]
    recording = d["recording"]

    cell_info = next(c for c in CHOSEN_CELLS if c[0] == session and c[1] == roi)
    why, depth_tuned = cell_info[2], cell_info[3]

    print(f"Fitting {session} ROI {roi} ({why}, depth_tuned={depth_tuned})...")
    start = time.time()
    row = fit_roi(X, y, recording, n_jobs=n_jobs)
    row.update(
        {
            "session": session,
            "roi": roi,
            "why": why,
            "depth_tuned": depth_tuned,
            "n_frames": int(len(y)),
        }
    )
    df = pd.DataFrame([row])
    df.to_pickle(target)
    print(f"Completed {session} ROI {roi} in {time.time() - start:.1f}s -> {target}")
    return df


def fit_all(out_dir, n_jobs=6):
    """Fit all 10 chosen cells."""
    for sess, roi, _, _ in CHOSEN_CELLS:
        fit_single_roi(sess, roi, out_dir, n_jobs=n_jobs)


def generate_summary(out_dir):
    """Compile summary tables and model selection statistics."""
    out_dir = Path(out_dir)
    rois_dir = out_dir / "rois"
    paths = [rois_dir / f"{sess}_{roi}.pickle" for sess, roi, _, _ in CHOSEN_CELLS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f"Warning: missing fits for {missing}")
        paths = [p for p in paths if p.exists()]

    df = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)

    # Compute gains relative to base and gain_offset
    m_names = list(MODELS.keys())
    for m in m_names:
        df[f"{m}_gain_over_base"] = df[f"{m}_test_rsq"] - df["base_test_rsq"]
        df[f"{m}_gain_over_gain_offset"] = (
            df[f"{m}_test_rsq"] - df["gain_offset_test_rsq"]
        )

    # Save summary dataframe
    df.to_pickle(out_dir / "summary_fits.pickle")
    df.to_csv(out_dir / "summary_fits.csv", index=False)

    print("\n" + "=" * 90)
    print("HELD-OUT TEST R-SQUARED ACROSS 10 CELLS")
    print("=" * 90)
    cols = ["session", "roi", "depth_tuned"] + [f"{m}_test_rsq" for m in m_names]
    sub = df[cols].copy()
    sub.columns = ["Session", "ROI", "Depth"] + m_names
    print(sub.to_string(index=False))

    print("\n" + "=" * 90)
    print("HELD-OUT TEST R-SQUARED GAIN OVER BASE (x 1000)")
    print("=" * 90)
    gain_cols = ["session", "roi"] + [
        f"{m}_gain_over_base" for m in m_names if m != "base"
    ]
    gain_df = df[gain_cols].copy()
    for col in gain_cols[2:]:
        gain_df[col] = (gain_df[col] * 1000).round(2)
    gain_df.columns = ["Session", "ROI"] + [m for m in m_names if m != "base"]
    print(gain_df.to_string(index=False))

    print("\n" + "=" * 90)
    print("BEST HELD-OUT MODEL PER CELL")
    print("=" * 90)
    test_rsq_cols = [f"{m}_test_rsq" for m in m_names]
    best_m = df[test_rsq_cols].idxmax(axis=1).str.replace("_test_rsq", "")
    df["best_test_model"] = best_m
    counts = best_m.value_counts()
    print("Winning model count (highest held-out R2):")
    print(counts.to_string())

    # Parsimony rule: simplest model within 10% margin of the best model's gain over base
    print("\n" + "=" * 90)
    print("PARSIMONY SELECTION (within 10% of max gain over base)")
    print("=" * 90)
    # Model complexity hierarchy: base (0) < gain, offset, tf (1) < gain_offset (2) < ...
    complexity = {
        "base": 0,
        "gain": 1,
        "offset": 1,
        "tf": 1,
        "gain_offset": 2,
        "gain_offset_tf": 3,
        "gain_offset_sf": 3,
        "gain_offset_speed": 3,
        "gain_offset_tf_sf": 4,
    }
    parsimonious_winners = []
    for _, row in df.iterrows():
        max_gain = max(row[f"{m}_gain_over_base"] for m in m_names)
        threshold = 0.90 * max_gain  # within 10% margin
        # candidate models meeting threshold
        qualifying = [m for m in m_names if row[f"{m}_gain_over_base"] >= threshold]
        # pick model with lowest complexity, break ties with gain
        qualifying.sort(key=lambda m: (complexity[m], -row[f"{m}_gain_over_base"]))
        chosen = qualifying[0]
        parsimonious_winners.append(chosen)

    df["parsimonious_model"] = parsimonious_winners
    print(pd.Series(parsimonious_winners).value_counts().to_string())

    # Write human-readable markdown table to results
    def df_to_md(d):
        headers = list(d.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, r in d.iterrows():
            vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    md_lines = [
        "# Running-Speed Model Selection Summary (10 V1 Cells)",
        "",
        "## Held-out Test R²",
        "",
        df_to_md(sub),
        "",
        "## Held-out Test R² Gain over Base (× 1000)",
        "",
        df_to_md(gain_df),
        "",
        "## Parsimony Selection Summary",
        "",
        "- **Winning model count (highest R²)**: "
        + ", ".join(f"{k}: {v}" for k, v in counts.items()),
        "- **Parsimonious model count**: "
        + ", ".join(
            f"{k}: {v}"
            for k, v in pd.Series(parsimonious_winners).value_counts().items()
        ),
        "",
    ]
    (out_dir / "MODEL_SELECTION_RESULTS.md").write_text("\n".join(md_lines))
    print(f"\nWritten {out_dir / 'MODEL_SELECTION_RESULTS.md'}")
    return df


def run_simulations(out_dir, n_jobs=6):
    """Sanity check on simulated data from 2 cells: verify parameter separability and recovery."""
    out_dir = Path(out_dir)
    sim_dir = out_dir / "simulations"
    sim_dir.mkdir(parents=True, exist_ok=True)

    sim_cells = [("PZAH10.2d_S20230920", 220), ("PZAH10.2f_S20230817", 544)]
    sim_results = []

    for sess, roi in sim_cells:
        frames_file = out_dir / "frames" / f"{sess}.npz"
        d = np.load(frames_file)
        (col,) = np.flatnonzero(d["rois"] == roi)
        X = d["X"]
        y_real = d["dff"][:, col]

        # Load fitted parameters
        roi_fit = pd.read_pickle(out_dir / "rois" / f"{sess}_{roi}.pickle").iloc[0]
        go_names = list(GratingParams._fields) + ["g", "b"]
        go_params = [roi_fit[f"gain_offset_{n}"] for n in go_names]

        # Real predictions and residual standard deviation
        pred_go = eval_model("gain_offset", X, *go_params)
        res_std = float(np.std(y_real - pred_go))

        np.random.seed(42)
        noise = np.random.normal(0, res_std, size=len(y_real))

        # --- Test 1: Ground truth is pure gain_offset (True a_tf = 0) ---
        y_sim_null = pred_go + noise
        fits_null = fit_models(X, y_sim_null, seed=101)
        est_atf_null = fits_null["gain_offset_tf"][0][12]

        # --- Test 2: Ground truth has synthetic TF shift (True a_tf = 0.5) ---
        true_atf = 0.5
        params_with_tf = list(go_params) + [true_atf]
        pred_tf = eval_model("gain_offset_tf", X, *params_with_tf)
        y_sim_shift = pred_tf + noise
        fits_shift = fit_models(X, y_sim_shift, seed=102)
        est_atf_shift = fits_shift["gain_offset_tf"][0][12]

        res = {
            "session": sess,
            "roi": roi,
            "noise_std": res_std,
            "null_true_a_tf": 0.0,
            "null_fitted_a_tf": float(est_atf_null),
            "shift_true_a_tf": true_atf,
            "shift_fitted_a_tf": float(est_atf_shift),
        }
        sim_results.append(res)
        print(f"Sim {sess} ROI {roi}:")
        print(f"  Null ground truth (true a_tf=0.0): fitted a_tf = {est_atf_null:.4f}")
        print(
            f"  Shift ground truth (true a_tf={true_atf:.1f}): fitted a_tf = {est_atf_shift:.4f}"
        )

    sim_df = pd.DataFrame(sim_results)
    sim_df.to_pickle(sim_dir / "simulations.pickle")
    sim_df.to_csv(sim_dir / "simulations.csv", index=False)
    print(f"Simulation results saved to {sim_dir / 'simulations.csv'}")
    return sim_df


def trial_level_data(X, y, recs):
    """Aggregate frame-level data into trial-level means over running frames."""
    stim_change = (
        (np.diff(X[0]) != 0)
        | (np.diff(X[1]) != 0)
        | (np.diff(X[2]) != 0)
        | (recs[1:] != recs[:-1])
    )
    trial_ids = np.zeros(len(recs), int)
    trial_ids[1:] = np.cumsum(stim_change)
    df_f = pd.DataFrame(
        {
            "trial": trial_ids,
            "log_sf": X[0],
            "log_tf": X[1],
            "dir": X[2],
            "r": X[3],
            "rec": recs,
            "y": y,
        }
    )
    agg = df_f.groupby("trial").agg(
        log_sf=("log_sf", "first"),
        log_tf=("log_tf", "first"),
        dir=("dir", "first"),
        r=("r", "mean"),
        rec=("rec", "first"),
        y=("y", "mean"),
    )
    X_tr = np.array(
        [agg.log_sf.values, agg.log_tf.values, agg["dir"].values, agg.r.values]
    )
    return X_tr, agg.y.values, agg.rec.values


def run_robustness(out_dir, n_jobs=6):
    """Run robustness checks on the 10 cells: trial-level, LAG 2, LAG 8, smoothed RS."""
    out_dir = Path(out_dir)
    rob_dir = out_dir / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)
    flm = flz.get_flexilims_session(PROJECT)

    results = []

    # 1. Trial-level robustness across all 10 cells
    print("\n--- Running robustness variant: trial_level ---")
    for sess, roi, why, depth_tuned in CHOSEN_CELLS:
        target = rob_dir / f"trial_level_{sess}_{roi}.pickle"
        if target.exists():
            row = pd.read_pickle(target).to_dict("records")[0]
        else:
            d = np.load(out_dir / "frames" / f"{sess}.npz")
            (col,) = np.flatnonzero(d["rois"] == roi)
            X_tr, y_tr, recs_tr = trial_level_data(
                d["X"], d["dff"][:, col], d["recording"]
            )
            fit_res = fit_roi(X_tr, y_tr, recs_tr, n_jobs=n_jobs)
            row = dict(
                variant="trial_level",
                session=sess,
                roi=roi,
                depth_tuned=depth_tuned,
                **fit_res,
            )
            pd.DataFrame([row]).to_pickle(target)
        results.append(row)
        print(
            f"  trial_level {sess} ROI {roi}: base R2={row['base_test_rsq']:.4f}, "
            f"gain R2={row['gain_test_rsq']:.4f}, "
            f"gain_offset R2={row['gain_offset_test_rsq']:.4f}, "
            f"gain_offset_tf R2={row['gain_offset_tf_test_rsq']:.4f}"
        )

    # 2. Frame variants (LAG 2, LAG 8, Smoothed RS)
    variants = [
        ("lag_2", {"lag": 2, "smooth_rs_sec": 0.0}),
        ("lag_8", {"lag": 8, "smooth_rs_sec": 0.0}),
        ("smoothed_rs_0.5s", {"lag": LAG, "smooth_rs_sec": 0.5}),
    ]

    cell_map = {}
    for sess, roi, why, depth_tuned in CHOSEN_CELLS:
        cell_map.setdefault(sess, []).append(roi)

    for var_name, kwargs in variants:
        print(f"\n--- Running robustness variant: {var_name} ---")
        for sess, rois in cell_map.items():
            X, dff, recs, _ = extract_session_frames(flm, sess, **kwargs)
            for roi in rois:
                col = roi
                target = rob_dir / f"{var_name}_{sess}_{roi}.pickle"
                if target.exists():
                    row = pd.read_pickle(target).to_dict("records")[0]
                else:
                    y = dff[:, col]
                    fit_res = fit_roi(X, y, recs, n_jobs=n_jobs)
                    row = dict(variant=var_name, session=sess, roi=roi, **fit_res)
                    pd.DataFrame([row]).to_pickle(target)
                results.append(row)
                print(
                    f"  {var_name} {sess} ROI {roi}: base R2={row['base_test_rsq']:.4f}, "
                    f"gain_offset R2={row['gain_offset_test_rsq']:.4f}, "
                    f"gain_offset_tf R2={row['gain_offset_tf_test_rsq']:.4f}"
                )

    rob_df = pd.DataFrame(results)
    rob_df.to_pickle(rob_dir / "robustness_variants.pickle")
    rob_df.to_csv(rob_dir / "robustness_variants.csv", index=False)
    print(f"Robustness results saved to {rob_dir / 'robustness_variants.csv'}")
    return rob_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["prepare", "fit", "summary", "simulate", "robustness"]
    )
    parser.add_argument("session", nargs="?", help="Session name (for single ROI fit)")
    parser.add_argument(
        "roi", nargs="?", type=int, help="ROI index (for single ROI fit)"
    )
    parser.add_argument(
        "--out-dir", required=True, help="Directory to save frames and fits"
    )
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel jobs per fit")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.out_dir)
    elif args.command == "fit":
        if args.session and args.roi is not None:
            fit_single_roi(args.session, args.roi, args.out_dir, n_jobs=args.n_jobs)
        else:
            fit_all(args.out_dir, n_jobs=args.n_jobs)
    elif args.command == "summary":
        generate_summary(args.out_dir)
    elif args.command == "simulate":
        run_simulations(args.out_dir, n_jobs=args.n_jobs)
    elif args.command == "robustness":
        run_robustness(args.out_dir, n_jobs=args.n_jobs)
