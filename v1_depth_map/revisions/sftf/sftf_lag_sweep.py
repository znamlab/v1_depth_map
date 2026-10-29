"""Response lag of the frame-level SF/TF fits of the gratings, for the best-fitted SF/TF-tuned cells.

The `base`, `rsg` and `rsg_tf` models of `precompute_data/sftf_running_models.py` (SF, TF and direction
tuning; + running-speed Gaussian on the amplitude; + TF preference shifted with running speed; warm-started as in
that script) are refitted with the dF/F (`ast_neuropil=False`) read `lag` imaging frames after the stimulus
and running speed it is paired with, as `sftf_running_models.session_frames` does for one lag. Running frames
only (RS > RS_THR), held-out R² leaving one recording out. Each session is synced once and every lag is cut
from the same arrays.

Cells (`select`): the N_CELLS SF/TF-tuned cells with the highest held-out R² of the trial-level SF/TF fit
(`fit_sftf_test_rsq` of `cache_gratings/grating_tuning_cells.csv`), a criterion with no frame lag, so the
selection does not favour any lag. Listed in CELLS_CSV, which travels with the code.

Steps, so that cells can be fitted as separate jobs (`sftf_lag_sweep.sbatch`):

    select               (locally) write CELLS_CSV
    prepare SESSION      sync once, save stimulus, RS and dF/F of the session's cells to OUT_DIR/frames/
    fit SESSION ROI      all lags of one cell -> OUT_DIR/rois/<session>_<roi>.pickle (skipped if it exists)
    merge                -> OUT_DIR/sftf_lag_sweep.csv, one row per cell and lag, with the parameters and
                         held-out R² of each model (`<model>_...` columns); copied to
                         `cache_gratings/lag_sweep/` of the figures folder (data drive), read by
                         `revisions/rf_spike_tau/lag_choice.ipynb`

Usage:
    python sftf_lag_sweep.py select [--n-cells 100]
    python sftf_lag_sweep.py {prepare,fit,merge} [SESSION] [ROI] --out-dir DIR [--lags=-2:15] [--n-jobs 1]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from v1_depth_map.precompute_data import sftf_running_models as srm

MODELS = ["base", "rsg", "rsg_tf"]  # rsg_tf and the models whose fits are saved
N_CELLS = 100
LAGS = "-2:15"  # frames, inclusive
CELLS_CSV = (
    Path(__file__).resolve().parents[1]
    / "rf_spike_tau"
    / "data"
    / "lag_sweep"
    / "sftf_lag_sweep_cells.csv"
)


def select(n_cells=N_CELLS):
    """Write the n_cells SF/TF-tuned cells with the best held-out trial-level SF/TF fit to CELLS_CSV."""
    import flexiznam as flz

    from v1_depth_map.paths import get_figures_roots

    flm = flz.get_flexilims_session(srm.PROJECT)
    _, root = get_figures_roots(flm, fig_subdir="cache_gratings")
    cells = pd.read_csv(root / "grating_tuning_cells.csv")
    cells = (
        cells[cells.sftf_tuned]
        .nlargest(n_cells, "fit_sftf_test_rsq")[
            ["session", "roi", "fit_sftf_test_rsq", "depth_tuned"]
        ]
        .sort_values(["session", "roi"])
    )
    cells.to_csv(CELLS_CSV, index=False)
    print(
        f"{len(cells)} cells, fit_sftf_test_rsq {cells.fit_sftf_test_rsq.min():.3f}-"
        f"{cells.fit_sftf_test_rsq.max():.3f} -> {CELLS_CSV}"
    )


def frames_path(out_dir, session):
    return Path(out_dir) / "frames" / f"{session}.npz"


def prepare(session, out_dir):
    """Stimulus, running speed and dF/F of the session's cells for each SFTF recording (unshifted)."""
    import flexiznam as flz

    from cottage_analysis.preprocessing import synchronisation

    flm = flz.get_flexilims_session(srm.PROJECT)
    cells = pd.read_csv(CELLS_CSV)
    rois = cells.roi[cells.session == session].to_numpy()
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()
    arrays = dict(rois=rois, names=recordings["name"].to_numpy())
    for i, (_, rec) in enumerate(recordings.iterrows()):
        vs_df = synchronisation.generate_vs_df(
            recording=rec,
            photodiode_protocol=srm.PHOTODIODE_PROTOCOL,
            flexilims_session=flm,
            project=srm.PROJECT,
        )
        img = synchronisation.generate_imaging_df(
            vs_df=vs_df,
            recording=rec,
            flexilims_session=flm,
            filter_datasets=srm.FILTER_TRACES,
            return_volumes=True,
        ).reset_index(drop=True)
        arrays[f"stim_{i}"] = img[srm.PARAMS].to_numpy(float)
        arrays[f"rs_{i}"] = (
            img.mouse_z_harp.diff() / img.mouse_z_harptime.diff()
        ).to_numpy()
        arrays[f"dff_{i}"] = np.vstack(img.dffs.values)[:, rois]
    path = frames_path(out_dir, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"{session}: {len(rois)} cells, {len(recordings)} recordings -> {path}")


def load_recordings(out_dir, session):
    d = np.load(frames_path(out_dir, session), allow_pickle=True)
    recs = [
        dict(name=name, stim=d[f"stim_{i}"], rs=d[f"rs_{i}"], dff=d[f"dff_{i}"])
        for i, name in enumerate(d["names"])
    ]
    return d["rois"], recs


def lagged_frames(recs, lag):
    """Running frames with the response `lag` frames after stimulus and RS (negative: before), as
    `sftf_running_models.session_frames`. Returns X (4, n), dF/F (n, n_rois), recording (n,) and the
    median log RS subtracted from X[3]."""
    xs, trs, names = [], [], []
    for r in recs:
        stim, rs, tr = r["stim"], r["rs"], r["dff"]
        n = len(stim) - abs(lag)
        if lag >= 0:
            stim, rs, tr = stim[:n], rs[:n], tr[lag:]
        else:
            stim, rs, tr = stim[-lag:], rs[-lag:], tr[:n]
        keep = np.isfinite(stim).all(axis=1) & np.isfinite(rs) & (rs > srm.RS_THR)
        xs.append(
            np.c_[np.log(stim[keep, :2]), np.deg2rad(stim[keep, 2]), np.log(rs[keep])]
        )
        trs.append(tr[keep])
        names.append(np.full(keep.sum(), r["name"]))
    X = np.concatenate(xs).T
    log_rs_median = np.median(X[3])
    X[3] -= log_rs_median
    return X, np.concatenate(trs), np.concatenate(names), log_rs_median


def parse_lags(text):
    lo, hi = (int(v) for v in text.split(":"))
    return list(range(lo, hi + 1))


def fit(session, roi, out_dir, lags, n_jobs):
    """Fit MODELS to one cell at every lag; skipped if the result exists.

    The lags done are saved after each one to <session>_<roi>.partial.pickle, so a job that runs out of time
    resumes from the next lag; the partial file is removed once all lags are done.
    """
    target = Path(out_dir) / "rois" / f"{session}_{roi}.pickle"
    if target.exists():
        print(f"{target} exists, skipping")
        return
    partial = target.with_suffix(".partial.pickle")
    rows = pd.read_pickle(partial).to_dict("records") if partial.exists() else []
    done = {r["lag"] for r in rows}
    rois, recs = load_recordings(out_dir, session)
    (col,) = np.flatnonzero(rois == roi)
    target.parent.mkdir(parents=True, exist_ok=True)
    for lag in lags:
        if lag in done:
            continue
        X, dff, names, log_rs_median = lagged_frames(recs, lag)
        row = srm.fit_roi(
            X,
            dff[:, col],
            names,
            n_jobs=n_jobs,
            models=MODELS,
            log_rs_median=log_rs_median,
        )
        rows.append(dict(session=session, roi=roi, lag=lag, n_frames=X.shape[1]) | row)
        pd.DataFrame(rows).to_pickle(partial)
        print(f"{session} ROI {roi} lag {lag}", flush=True)
    pd.DataFrame(rows).sort_values("lag").to_pickle(target)
    partial.unlink()


def merge(out_dir):
    """Concatenate the cells of CELLS_CSV; fails if any is missing."""
    out_dir = Path(out_dir)
    cells = pd.read_csv(CELLS_CSV)
    paths = [
        out_dir / "rois" / f"{s}_{r}.pickle" for s, r in zip(cells.session, cells.roi)
    ]
    missing = [p.name for p in paths if not p.exists()]
    assert not missing, f"{len(missing)} cells missing, e.g. {missing[:5]}"
    out = pd.concat([pd.read_pickle(p) for p in paths], ignore_index=True)
    out.to_csv(out_dir / "sftf_lag_sweep.csv", index=False)
    print(f"Saved {len(out)} rows to {out_dir / 'sftf_lag_sweep.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["select", "prepare", "fit", "merge"])
    p.add_argument("session", nargs="?")
    p.add_argument("roi", nargs="?", type=int)
    p.add_argument("--out-dir")
    p.add_argument("--lags", default=LAGS, help="lo:hi frames, inclusive")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--n-cells", type=int, default=N_CELLS)
    a = p.parse_args()
    if a.command == "select":
        select(a.n_cells)
    elif a.command == "prepare":
        prepare(a.session, a.out_dir)
    elif a.command == "fit":
        fit(a.session, a.roi, a.out_dir, parse_lags(a.lags), a.n_jobs)
    else:
        merge(a.out_dir)
