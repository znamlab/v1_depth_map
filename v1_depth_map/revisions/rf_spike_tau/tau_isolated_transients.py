"""Calcium decay time constant from the average of isolated transients, per cell.

This is how the deconvolution tau (`deconvolve_tau.py`) was chosen: 1.1 s for the
GCaMP6s mice (PZAG*) and 0.25 s for the GCaMP6f mice (PZAH*).

For each cell (`iscell`) of the concatenated suite2p dataset whose split indexing gives
the analysed traces (`concatenated_dataset`):
1. `Fstandard.npy` (neuropil-corrected F, the input of the deconvolution) minus a
   maximin baseline (10-frame boxcar, min then max filter over 60 s), divided by the
   median baseline.
2. Transients are detected on the trace smoothed over 3 frames: peaks above the median
   + 6 robust SD (1.4826 MAD) with a prominence above 4 SD. A transient is isolated if
   no other detected peak is within 4 s before or after it.
3. Cells with at least 5 isolated transients: the median of the unsmoothed trace from
   0.5 s before to 4 s after each peak (event-triggered average) is fitted with
   a exp(-t / tau) + c from one frame after the peak (tau bounded to 0.05-10 s).

Two other estimates were tried and discarded: an exponential fit of each transient
from its peak (most fits at the bounds, tau 0.05-0.7 s) and the decay of the trace
autocovariance at lags >= 1 frame (2.5-4.9 s for GCaMP6s, inflated by slow
fluctuations of the trace).

Saves, per session, the per-cell fits and event-triggered averages, and example traces.

Needs the `2p-preprocess` environment (suite2p, via `deconvolve_tau`).

Usage:
    python tau_isolated_transients.py SESSION --out DIR [--project hey2_3d-vision_foodres_20220101]
"""

import argparse
from pathlib import Path

import numpy as np
import flexiznam as flz
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from v1_depth_map.precompute_data.deconvolve_tau import (
    PROJECT,
    TRACES,
    concatenated_dataset,
)

PRE, POST = 0.5, 4.0  # event-triggered average window (s)
ISOLATION = 4.0  # no other peak within this time (s)
MIN_EVENTS = 5
N_EXAMPLES = 3
EXAMPLE_WINDOW = 120.0  # s


def normalised_trace(F, fs):
    """(F - maximin baseline) / median baseline, as in suite2p's maximin (60 s window)."""
    win = int(60 * fs)
    base = maximum_filter1d(
        minimum_filter1d(uniform_filter1d(F, 10, axis=1), win, axis=1), win, axis=1
    )
    return (F - base) / np.maximum(np.median(base, axis=1, keepdims=True), 1e-6)


def isolated_events(xs, fs):
    """Isolated transients of one smoothed trace, away from the edges of the window."""
    sd = 1.4826 * np.median(np.abs(xs - np.median(xs)))
    pk, _ = find_peaks(xs, height=np.median(xs) + 6 * sd, prominence=4 * sd)
    gap = ISOLATION * fs
    iso = pk[(np.diff(np.r_[-1e9, pk]) > gap) & (np.diff(np.r_[pk, 1e9]) > gap)]
    return iso[(iso > int(PRE * fs)) & (iso < len(xs) - int(POST * fs))]


def exp_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def fit_plane(F, iscell, fs):
    """Per-cell fits and event-triggered averages of one plane."""
    x = normalised_trace(F.astype(np.float64), fs)
    xs = uniform_filter1d(x, 3, axis=1)
    pre, post = int(PRE * fs), int(POST * fs)
    t = np.arange(1, post) / fs
    out = dict(roi=[], n_events=[], tau=[], amp=[], offset=[], eta=[], events=[])
    for i in np.flatnonzero(iscell):
        iso = isolated_events(xs[i], fs)
        popt = np.full(3, np.nan)
        eta = np.full(pre + post, np.nan)
        if len(iso) >= MIN_EVENTS:
            eta = np.median(np.stack([x[i, k - pre : k + post] for k in iso]), axis=0)
            seg = eta[pre + 1 :]
            try:
                popt, _ = curve_fit(
                    exp_decay,
                    t,
                    seg,
                    p0=[seg[0], 1, 0],
                    bounds=([0, 0.05, -np.inf], [np.inf, 10, np.inf]),
                    maxfev=5000,
                )
            except RuntimeError:
                pass
        out["roi"].append(i)
        out["n_events"].append(len(iso))
        out["amp"].append(popt[0])
        out["tau"].append(popt[1])
        out["offset"].append(popt[2])
        out["eta"].append(eta)
        out["events"].append(iso)
    return x, out


def examples(x, res, fs):
    """Traces of the cells closest to the median tau (>= 10 events), around their events."""
    tau = np.array(res["tau"])
    n = np.array(res["n_events"])
    ok = np.flatnonzero(np.isfinite(tau) & (n >= 10))
    if len(ok) == 0:
        return {}
    pick = ok[np.argsort(np.abs(tau[ok] - np.median(tau[ok])))[:N_EXAMPLES]]
    width = int(EXAMPLE_WINDOW * fs)
    traces, starts, events = [], [], []
    for j in pick:
        ev = res["events"][j]
        # window starting 10 s before the event with the most isolated events in the window
        counts = [np.sum((ev >= e - 10 * fs) & (ev < e - 10 * fs + width)) for e in ev]
        start = int(max(0, min(ev[np.argmax(counts)] - 10 * fs, x.shape[1] - width)))
        traces.append(x[res["roi"][j], start : start + width])
        starts.append(start)
        events.append(ev[(ev >= start) & (ev < start + width)] - start)
    return dict(
        example_roi=np.array(res["roi"])[pick],
        example_trace=np.array(traces, dtype=np.float32),
        example_start=np.array(starts),
        example_events=np.array(
            [np.pad(e, (0, width - len(e)), constant_values=-1) for e in events]
        ),
    )


def main(session, out, project=PROJECT):
    fs_flz = flz.get_flexilims_session(project)
    ds, _ = concatenated_dataset(fs_flz, session, project)
    nplanes = int(float(ds.extra_attributes["nplanes"]))
    planes = []
    for iplane in range(nplanes):
        p = ds.path_full / f"plane{iplane}"
        F = np.load(p / "Fstandard.npy")
        if F.shape[0] == 0:
            continue
        ops = np.load(p / "ops.npy", allow_pickle=True).item()
        fs = float(ops["fs"])
        iscell = np.load(p / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
        x, res = fit_plane(F, iscell, fs)
        res["plane"] = [iplane] * len(res["roi"])
        res["fs"] = fs
        res.update(examples(x, res, fs))
        planes.append(res)
        tau = np.array(res["tau"])
        v = tau[np.isfinite(tau)]
        print(
            f"{session} plane {iplane}: {fs:.2f} Hz, {F.shape[1]} frames, {iscell.sum()} cells, "
            f"{len(v)} with >= {MIN_EVENTS} isolated events: tau median {np.median(v):.2f} s, "
            f"IQR {np.quantile(v, 0.25):.2f}-{np.quantile(v, 0.75):.2f}, at bounds "
            f"{np.mean((v < 0.06) | (v > 9.9)):.0%}"
        )
    # examples from the plane with most fitted cells
    best = max(planes, key=lambda r: np.isfinite(r["tau"]).sum())

    def cat(k):
        return np.concatenate([np.asarray(r[k]) for r in planes])

    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Path(out) / f"{session}_tau.npz",
        session=session,
        project=project,
        indicator="GCaMP6s" if session.startswith("PZAG") else "GCaMP6f",
        fs=best["fs"],
        plane=cat("plane"),
        roi=cat("roi"),
        n_events=cat("n_events"),
        tau=cat("tau"),
        amp=cat("amp"),
        offset=cat("offset"),
        eta=np.concatenate([np.array(r["eta"]) for r in planes]).astype(np.float32),
        eta_t=(
            np.arange(int(PRE * best["fs"]) + int(POST * best["fs"]))
            - int(PRE * best["fs"])
        )
        / best["fs"],
        example_plane=best["plane"][0],
        **{k: best[k] for k in best if k.startswith("example_")},
    )
    print(f"saved {Path(out) / f'{session}_tau.npz'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--out", required=True)
    p.add_argument("--project", default=PROJECT, choices=sorted(TRACES))
    a = p.parse_args()
    main(a.session, a.out, a.project)
