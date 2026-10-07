"""Summarise the RF lag sweeps into one small table (used by lag_choice.ipynb).

Inputs are the outputs of `latency_sweep.py` (multidepth, `--model snapshot`) and
`single_depth_lag_sweep.py`, on dF/F and on spikes (`--use-col spks`), possibly split
over several files per session (one per group of lags):
    DFF_MD: <session>_lags_*.npz          SPK_MD: <session>_md_spk_lags.npz
    DFF_SD: <session>_sd_lags_*.npz       SPK_SD: <session>_sd_spk_lags.npz

Cells: those with a best R2 > 0 over lags. Multidepth sweeps were run on cells
with R2 > 0.05 in a previous fit; single depth cells must have a pipeline
single-depth R2 (`stored_r2`) > 0.03.

For each session, recording type, signal and lag, the table gives the number of
cells, the median over cells of R2 / the cell's best R2 over lags, the median R2,
and the fraction of cells whose best lag it is.

Usage:
    python summarise_lag_sweeps.py --dff-md DIR --dff-sd DIR --spikes DIR --out FILE.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SD_MIN_R2 = 0.03


def load(files, stored_cut=None):
    """Lags (frames) and per-cell best R2 (cells x lags) merged over files."""
    parts = [np.load(f) for f in sorted(files)]
    fp = float(parts[0]["frame_period"])
    lag = np.concatenate([np.round(p["latencies"] / fp) for p in parts]).astype(int)
    r2 = np.concatenate([p["best_r2"] for p in parts], axis=1)
    lag, u = np.unique(lag, return_index=True)
    r2 = r2[:, u]
    ok = np.nanmax(r2, axis=1) > 0
    if stored_cut is not None:
        ok &= parts[0]["stored_r2"] > stored_cut
    return lag, fp, r2[ok]


def summarise(session, recording, signal, lag, fp, r2):
    norm = r2 / np.nanmax(r2, axis=1, keepdims=True)
    best = lag[np.nanargmax(r2, axis=1)]
    return pd.DataFrame(
        dict(
            session=session,
            indicator="GCaMP6s" if session.startswith("PZAG") else "GCaMP6f",
            recording=recording,
            signal=signal,
            lag_frames=lag,
            lag_ms=np.round(lag * fp * 1000, 1),
            n_cells=len(r2),
            median_r2_rel_best=np.nanmedian(norm, axis=0),
            median_r2=np.nanmedian(r2, axis=0),
            frac_best_lag=[np.mean(best == k) for k in lag],
        )
    )


def main(dff_md, dff_sd, spikes, out):
    sources = [
        ("multidepth", "dff", Path(dff_md), "_lags_*.npz", None),
        ("single_depth", "dff", Path(dff_sd), "_sd_lags_*.npz", SD_MIN_R2),
        ("multidepth", "spikes", Path(spikes), "_md_spk_lags.npz", None),
        ("single_depth", "spikes", Path(spikes), "_sd_spk_lags.npz", SD_MIN_R2),
    ]
    tables = []
    for recording, signal, folder, suffix, cut in sources:
        sessions = sorted(
            {f.name.split(suffix.split("*")[0])[0] for f in folder.glob("*" + suffix)}
        )
        for s in sessions:
            lag, fp, r2 = load(folder.glob(s + suffix), cut)
            tables.append(summarise(s, recording, signal, lag, fp, r2))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out, index=False, float_format="%.4g")
    print(table.groupby(["recording", "signal"]).session.nunique())
    print(f"saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dff-md", required=True)
    p.add_argument("--dff-sd", required=True)
    p.add_argument("--spikes", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    main(a.dff_md, a.dff_sd, a.spikes, a.out)
