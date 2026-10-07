"""Diagnostics of the deconvolution tau that do not need an RF fit.

For spikes from `deconvolve_tau.py --out-dir`, the baseline-corrected trace (the
input of OASIS) is compared with the spikes convolved with exp(-t / tau):
- variance explained by the reconvolved spikes (fit gain and offset per ROI),
- autocorrelation of the residual: a slow tail means the kernel misses slow
  components, i.e. tau is too short,
- autocorrelation of the spikes: a slow tail also means tau too short (decay left
  in the spikes).

Needs the `2p-preprocess` environment (suite2p).

Usage:
    python tau_diagnostics.py SESSION --dirs DIR1,DIR2 --taus 0.7,1.0 --out FILE.npz
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from suite2p.extraction import dcnv

import flexiznam as flz

from v1_depth_map.precompute_data.deconvolve_tau import PROJECT, concatenated_dataset

MAX_LAG = 90  # frames, about 6 s


def autocorr(x, max_lag):
    """Mean autocorrelation over ROIs, lags 0..max_lag. x: (nrois, nframes)."""
    x = x - x.mean(axis=1, keepdims=True)
    var = np.sum(x**2, axis=1)
    ok = var > 0
    x, var = x[ok], var[ok]
    return np.array(
        [
            np.mean(np.sum(x[:, : x.shape[1] - k] * x[:, k:], axis=1) / var)
            for k in range(max_lag + 1)
        ]
    )


def main(session, dirs, taus, out):
    fs_flz = flz.get_flexilims_session(PROJECT)
    ds, _ = concatenated_dataset(fs_flz, session)
    plane = ds.path_full / "plane0"
    ops = np.load(plane / "ops.npy", allow_pickle=True).tolist()
    fs = ops["fs"]
    F = dcnv.preprocess(
        F=np.load(plane / "Fstandard.npy"),
        baseline=ops["baseline_method"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=fs,
    )
    good = np.all(np.isfinite(F), axis=1) & (np.std(F, axis=1) > 0)
    F = F[good]
    res = dict(taus=np.array(taus), lags_s=np.arange(MAX_LAG + 1) / fs, fs=fs)
    r2, ac_res, ac_spk = [], [], []
    for d, tau in zip(dirs, taus):
        S = np.load(Path(d) / "concatenated" / "plane0" / "spks.npy")[good]
        g = np.exp(-1 / (tau * fs))
        C = lfilter([1.0], [1.0, -g], S, axis=1)  # calcium = AR(1) of the spikes
        # least-squares gain and offset per ROI
        Cm, Fm = C - C.mean(1, keepdims=True), F - F.mean(1, keepdims=True)
        gain = np.sum(Cm * Fm, 1) / np.maximum(np.sum(Cm**2, 1), 1e-12)
        resid = Fm - gain[:, None] * Cm
        r2.append(1 - np.sum(resid**2, 1) / np.sum(Fm**2, 1))
        ac_res.append(autocorr(resid, MAX_LAG))
        ac_spk.append(autocorr(S, MAX_LAG))
        print(
            f"tau {tau}: median R2 reconvolved vs F {np.median(r2[-1]):.3f}; residual autocorr at "
            f"0.5/1/2 s {np.interp([0.5, 1, 2], res['lags_s'], ac_res[-1]).round(3).tolist()}; spike autocorr "
            f"{np.interp([0.5, 1, 2], res['lags_s'], ac_spk[-1]).round(3).tolist()}"
        )
    np.savez(
        out, r2=np.array(r2), ac_res=np.array(ac_res), ac_spk=np.array(ac_spk), **res
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--dirs", required=True)
    p.add_argument("--taus", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.session, a.dirs.split(","), [float(t) for t in a.taus.split(",")], a.out)
