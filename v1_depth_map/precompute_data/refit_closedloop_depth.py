"""Refit the closed-loop 1D depth tuning of one revision session, then merge it.

Fit step: same fits as the depth block of `analysis_pipeline.main` for
`SpheresPermTubeReward` (find_depth_neurons, then fit_preferred_depth on all trials, on
odd trials and with 5-fold cross-validation, for all frames, running frames and stationary
frames). The session's `neurons_df.pickle` is only read (for ROI alignment); the
closed-loop depth columns are saved, with `roi`, to
OUT_DIR/SESSION/closedloop_depth_fit.pickle, with provenance in a .json.

Merge step (`--merge`): copies every column of that result (including the `_running` /
`_notrunning` fits, which this refit redoes) into the session's `neurons_df.pickle`
(located through flexiznam, as in the fit step), with the backup, atomic write and
verification of `refit_treadmill_depth.merge_into_neurons_df`.

Usage:
    python refit_closedloop_depth.py SESSION --out-dir DIR [--overwrite]
    python refit_closedloop_depth.py SESSION --out-dir DIR --merge [--dry-run]
        [--backup-suffix SUFFIX] [--overwrite-backup]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import flexiznam as flz

import cottage_analysis
from cottage_analysis.analysis import common_utils, find_depth_neurons
from cottage_analysis.pipelines import pipeline_utils
from v1_depth_map.precompute_data.refit_treadmill_depth import (
    DEPTH_COLUMN_KEYS,
    PHOTODIODE_PROTOCOL,
    PROJECT,
    add_merge_arguments,
    code_version,
    get_neurons_ds,
    merge_into_neurons_df,
)

FILTER_ROIS = {"anatomical_only": 3, "annotated": True}
DEPTH_FIT_PARAMS = dict(depth_min=0.02, depth_max=20, niter=10, min_sigma=0.5)
# (rs_thr, rs_thr_max, still_only, still_time, special_sfx), as in analysis_pipeline
CONDITIONS = [
    (None, None, False, 0, ""),
    (0.05, None, False, 0, "_running"),
    (None, 0.05, True, 1, "_notrunning"),
]


def result_path(out_dir, session):
    return Path(out_dir) / session / "closedloop_depth_fit.pickle"


def main(session, out_dir, overwrite=False):
    out = result_path(out_dir, session)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} exists: not overwriting (pass --overwrite)")

    flm = flz.get_flexilims_session(project_id=PROJECT)
    neurons_ds = get_neurons_ds(session, flm)
    neurons_df = pd.read_pickle(neurons_ds.path_full)  # read only
    suite2p_ds = flz.get_datasets(
        origin_name=session,
        dataset_type="suite2p_rois",
        flexilims_session=flm,
        filter_datasets=FILTER_ROIS,
        allow_multiple=False,
        return_dataseries=False,
    )
    frame_rate = suite2p_ds.extra_attributes["fs"]

    t0 = time.time()
    _, _, _, trials_df = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets=FILTER_ROIS,
        protocol_base="SpheresPermTubeReward",
        recording_type="two_photon",
    )
    trials_df = trials_df[~trials_df.recording_name.str.contains("multidepth")]
    print(
        f"{session}: loaded {len(trials_df)} trials in {time.time() - t0:.0f} s",
        flush=True,
    )

    ndf, nds = find_depth_neurons.find_depth_neurons(
        trials_df=trials_df,
        neurons_ds=neurons_ds,
        neurons_df=neurons_df.copy(),
        rs_thr=None,
        alpha=0.05,
        special_sfx="",
        max_rs2motor_diff=None,
    )
    for rs_thr, rs_thr_max, still_only, still_time, sfx in CONDITIONS:
        for k_folds, choose_trials in [(1, None), (1, "odd"), (5, None)]:
            t1 = time.time()
            ndf, nds = find_depth_neurons.fit_preferred_depth(
                trials_df=trials_df,
                neurons_df=ndf,
                neurons_ds=nds,
                closed_loop=1,
                choose_trials=choose_trials,
                rs_thr=rs_thr,
                rs_thr_max=rs_thr_max,
                still_only=still_only,
                still_time=still_time,
                frame_rate=frame_rate,
                k_folds=k_folds,
                special_sfx=sfx,
                max_rs2motor_diff=None,
                **DEPTH_FIT_PARAMS,
            )
            print(
                f"{session}: fit{sfx} k_folds={k_folds} trials={choose_trials} "
                f"in {time.time() - t1:.0f} s",
                flush=True,
            )
    common_utils.add_one_sided_spearman_significance(
        ndf,
        rval_col="depth_tuning_test_spearmanr_rval_closedloop",
        pval_col="depth_tuning_test_spearmanr_pval_closedloop",
        out_col="is_depth_neuron",
    )

    cols = [
        c
        for c in ndf.columns
        if any(key in c for key in DEPTH_COLUMN_KEYS) and "treadmill" not in c
    ]
    result = ndf[["roi"] + cols].copy()
    assert result.roi.equals(neurons_df.roi), "ROI order changed"
    result.to_pickle(out)

    old_sel = neurons_df.get("is_depth_neuron")
    provenance = dict(
        session=session,
        protocol_base="SpheresPermTubeReward",
        depth_fit_params=DEPTH_FIT_PARAMS,
        frame_rate=float(frame_rate),
        n_trials=int(len(trials_df)),
        neurons_df=str(neurons_ds.path_full),
        cottage_analysis=code_version(cottage_analysis),
        columns=list(result.columns),
        n_depth_selective_new=int(result["is_depth_neuron"].sum()),
        n_depth_selective_old=None if old_sel is None else int(old_sel.sum()),
        runtime_s=round(time.time() - t0),
    )
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


def merge(session, out_dir, **kwargs):
    """Merge every column of OUT_DIR/SESSION/closedloop_depth_fit.pickle into neurons_df."""
    return merge_into_neurons_df(
        get_neurons_ds(session).path_full, result_path(out_dir, session), **kwargs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    add_merge_arguments(parser)
    args = parser.parse_args()
    if not args.merge and (args.dry_run or args.backup_suffix or args.overwrite_backup):
        parser.error("--dry-run, --backup-suffix and --overwrite-backup need --merge")
    if args.merge:
        merge(
            args.session,
            args.out_dir,
            backup_suffix=args.backup_suffix,
            dry_run=args.dry_run,
            overwrite_backup=args.overwrite_backup,
        )
    else:
        main(args.session, args.out_dir, overwrite=args.overwrite)
