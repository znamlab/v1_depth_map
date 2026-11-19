"""Running and slow imaging frames of the drifting-grating recordings, for the speed-curve panels of Fig. S10.

For the drifting grating analysis (`figures/figsupp10_gratings.ipynb`, panels K-P). Synchronises every SFTF
recording of a session once and saves, per imaging frame during gratings, the stimulus (log SF, log TF, direction)
and the running speed, with the traces of the session's iscell ROIs:

- running frames: RS > RS_THR (used only when log-transforming to cut zeros, not as a definition of static), saved to
  OUT_DIR/frames/<session>.npz; the RS column of X is log RS minus the median log RS of the session;
- with --slow, the frames with RS < SLOW_THR (2.83 cm/s, the low edge of the slowest motor-speed bin of the figure 3
  treadmill, and of the slowest running bin of the figure), whatever the running speed before or after, saved to
  OUT_DIR/frames_slow/<session>.npz; the RS column of X is 0.

The neural trace is shifted by LAG frames relative to the stimulus and RS (default 0 for dF/F, as the RS/OF fits;
2 for deconvolved spikes). The first SKIP frames of each grating are dropped, as the trial window of the trial-level
analyses (`[onset + 8, next_onset)`, `gratings.generate_trials_df`): they carry the decay of the previous grating's
response. The figure reads the frames with `sftf.build_speed_curves`; its `RELOAD` (`sftf.rebuild_gratings_cache`)
prepares them under the analysis root with `FIGURE_FRAMES`, the defaults below.

Usage:
    python sftf_frames.py SESSION --out-dir DIR [--slow] [--trace {dff,spks}] [--lag LAG] [--skip SKIP]
"""

import argparse
from pathlib import Path

import flexiznam as flz
import numpy as np

from cottage_analysis.preprocessing import synchronisation

PROJECT = "hey2_3d-vision_foodres_20220101"
PHOTODIODE_PROTOCOL = 5
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}
PARAMS = ["SpatialFrequency", "TemporalFrequency", "Angle"]
RS_THR = 0.01  # m/s, floor used only when log-transforming RS to cut 0s (as rs_thr of the RS/OF fits; not a definition of static)
# m/s, max RS of the slow frames (--slow): low edge of the slowest treadmill bin of figure 3 (2.83 cm/s)
SLOW_THR = 2**1.5 / 100
LAG = 0  # default frames between stimulus / RS and signal for dF/F (as the RS/OF fits; 2 for deconvolved spikes)
SKIP = 8  # frames dropped after each grating onset (0.53 s at 15.2 Hz, as the trial window of the trial-level fits)
# settings of the figure (figsupp10_gratings): dF/F at lag 0 as the RS/OF fits
FIGURE_FRAMES = dict(trace="dff", lag=0, skip=SKIP)


def session_frames(flm, session, trace="dff", lag=LAG, slow=False, skip=SKIP):
    """Running (or slow) frames during gratings of all SFTF recordings of a session.

    Args:
        flm: Flexilims session.
        session: session name.
        trace: "spks" (deconvolved spikes) or "dff" (dF/F).
        lag: number of frames to shift signal relative to stimulus and RS.
        slow: keep the frames with RS < SLOW_THR (whatever the RS of the other frames) instead of the
            running frames.
        skip: drop the frames whose stimulus frame is fewer than `skip` frames after the onset of its grating
            (onset = change of the parameter-log time, as the trial tables).

    Returns:
        X (4, n_frames): log SF, log TF, direction (rad), log RS - median log RS (0 if `slow`);
        traces (n_frames, n_rois), recording (n_frames,), median log RS (NaN if `slow`).
    """
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=flm
    )
    recordings = recordings[recordings["name"].str.contains("SFTF")].sort_index()
    xs, traces_list, recs = [], [], []
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
            add_spikes=(trace == "spks"),
        ).reset_index(drop=True)
        rs = (img.mouse_z_harp.diff() / img.mouse_z_harptime.diff()).to_numpy()
        stim = img[PARAMS].to_numpy(float)
        tr = np.vstack(img.spks.values if trace == "spks" else img.dffs.values)
        # frames since the onset of the current grating
        harptime = img["stimulus_harptime"]
        onset = (harptime.notna() & (harptime != harptime.shift())).to_numpy()
        frame = np.arange(len(img))
        since_onset = frame - np.maximum.accumulate(np.where(onset, frame, 0))
        # stimulus and RS of frame t with signal of frame t + lag
        n = len(stim) - lag  # not stim[:-lag], empty for lag 0
        stim, rs, tr = stim[:n], rs[:n], tr[lag:]
        after_onset = since_onset[:n] >= skip
        if slow:
            keep = (
                np.isfinite(stim).all(axis=1)
                & after_onset
                & np.isfinite(rs)
                & (rs < SLOW_THR)
            )
            log_rs = np.zeros(keep.sum())
        else:
            keep = (
                np.isfinite(stim).all(axis=1)
                & after_onset
                & np.isfinite(rs)
                & (rs > RS_THR)
            )
            log_rs = np.log(rs[keep])
        xs.append(np.c_[np.log(stim[keep, :2]), np.deg2rad(stim[keep, 2]), log_rs])
        traces_list.append(tr[keep])
        recs.append(np.full(keep.sum(), rec["name"]))
    X = np.concatenate(xs).T
    log_rs_median = np.nan if slow else np.median(X[3])
    if not slow:
        X[3] -= log_rs_median
    return X, np.concatenate(traces_list), np.concatenate(recs), log_rs_median


def iscell_rois(flm, session):
    """Boolean iscell mask of the suite2p ROIs of a session."""
    rois_ds = flz.get_datasets(
        origin_name=session,
        dataset_type="suite2p_rois",
        filter_datasets={"anatomical_only": 3},
        flexilims_session=flm,
        allow_multiple=False,
    )
    return np.load(rois_ds.path_full / "plane0" / "iscell.npy")[:, 0] == 1


def frames_path(out_dir, session, slow=False):
    return Path(out_dir) / ("frames_slow" if slow else "frames") / f"{session}.npz"


def prepare(session, out_dir, trace="dff", lag=LAG, slow=False, skip=SKIP):
    """Sync the session once and save the running (or, with `slow`, slow) frames and the iscell ROIs' signal."""
    flm = flz.get_flexilims_session(PROJECT)
    rois = np.flatnonzero(iscell_rois(flm, session))
    X, traces, recording, log_rs_median = session_frames(
        flm, session, trace=trace, lag=lag, slow=slow, skip=skip
    )
    path = frames_path(out_dir, session, slow=slow)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        X=X,
        dff=traces[:, rois],  # the trace named by `trace` (dF/F or spikes)
        rois=rois,
        recording=recording,
        log_rs_median=log_rs_median,
        trace=trace,
        lag=lag,
        slow=slow,
        slow_thr=SLOW_THR,
        skip=skip,
    )
    print(
        f"{session} ({trace}, lag={lag}, skip={skip}): {X.shape[1]} {'slow' if slow else 'running'} frames, {len(rois)} ROIs -> {path}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--trace",
        choices=["dff", "spks"],
        default="dff",
        help="trace type to extract (default: dff)",
    )
    p.add_argument(
        "--lag",
        type=int,
        default=None,
        help=f"stimulus to trace lag in frames (default: {LAG} for dff, 2 for spks)",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=SKIP,
        help=f"frames dropped after each grating onset (default: {SKIP})",
    )
    p.add_argument(
        "--slow",
        action="store_true",
        help="frames below SLOW_THR instead of running frames, saved to frames_slow/",
    )
    a = p.parse_args()
    lag = a.lag if a.lag is not None else (2 if a.trace == "spks" else LAG)
    prepare(a.session, a.out_dir, trace=a.trace, lag=lag, slow=a.slow, skip=a.skip)
