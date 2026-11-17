"""RS/OF fits of one session and target as a Slurm array over batches of ROIs.

Same fits as `fit_revision_treadmill.py` (targets `sphere`, `treadmill`,
`treadmill_frames`; the 11 model configurations of the analysis pipeline; same fit
parameters and file names), but:

- the session is loaded once, then every (configuration, ROI batch) is one 1-CPU array
  task (`cottage_analysis.pipelines.fit_rsof_array`). Each ROI is fitted independently,
  so the results are identical to the unbatched fits;
- the merged `fit_rs_of_tuning_*.pickle` files go to OUT_DIR/SESSION/, never to the
  session folder: nothing in production is written. A `rsof_batched_{target}.json` next
  to them records the parameters and code version.

Run it inside a Slurm job (it loads the session), from a run directory whose `src/` is on
PYTHONPATH; the array and merge jobs it submits inherit PYTHONPATH.

Usage:
    python refit_rsof_batched.py SESSION TARGET --out-dir DIR --work-root DIR
        [--method plateau] [--roi-batch-size 25] [--time 3:00:00] [--mem 4G]
        [--max-concurrent 200] [--prepare-only]
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

import cottage_analysis
from cottage_analysis.io_module.visstim import get_frame_log
from cottage_analysis.pipelines import fit_rsof_array, pipeline_utils
from v1_depth_map.precompute_data.fit_revision_treadmill import (
    COMMON_PARAMS,
    MODEL_CONFIGS,
    PHOTODIODE_PROTOCOL,
    PROJECT,
    fit_filename,
    target_config,
)

# Traces used by all fits of the revision project (annotated = curated Cellpose masks)
FILTER_DATASETS = {"anatomical_only": 3, "ast_neuropil": False, "annotated": True}


def code_version(module):
    """Commit of the checkout a module is imported from, or the run directory record."""
    root = Path(module.__file__).resolve().parents[1]
    record = root.parent / "CODE_VERSION.txt"
    if record.exists():
        return record.read_text().strip()
    out = subprocess.run(
        ["git", "-C", str(root), "describe", "--always", "--dirty"],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() or "unknown"


def frame_index_offsets(session, protocol_base):
    """First FrameIndex of the frame log of each recording of the protocol.

    A non-zero value is what the FrameIndex bug of `generate_vs_df` got wrong; it is
    recorded so that the run shows which recordings the fix changes.
    """
    fs = flz.get_flexilims_session(project_id=PROJECT)
    recordings = flz.get_children(
        parent_name=session, children_datatype="recording", flexilims_session=fs
    )
    out = {}
    for name in recordings.name:
        if protocol_base not in name:
            continue
        try:
            rec = flz.get_entity(name=name, flexilims_session=fs)
            frame_log = get_frame_log(fs, harp_recording=rec, vis_stim_recording=rec)
            out[name] = int(frame_log.FrameIndex.iloc[0])
        except Exception as err:  # the record is informative only
            out[name] = f"not read: {err}"
    return out


def main(
    session,
    target,
    out_dir,
    work_root,
    method="plateau",
    roi_batch_size=25,
    time_limit="3:00:00",
    mem="4G",
    max_concurrent=200,
    prepare_only=False,
):
    cfg = target_config(target, method)
    out_dir = Path(out_dir) / session
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(work_root) / f"{session}_{target}"
    is_treadmill = cfg["protocol_base"] == "SpheresTubeMotor"
    tread_kwargs = {"method": method} if is_treadmill else None

    fit_params = dict(
        trial_sfx="",
        max_rs2motor_diff=cfg["max_rs2motor_diff"],
        trial_average=cfg["trial_average"],
        param_range=cfg["param_range"],
        use_col="dffs",
        **COMMON_PARAMS,
    )
    neurons_ds, trials_df = pipeline_utils.load_trials_for_rsof_fit(
        PROJECT,
        session,
        PHOTODIODE_PROTOCOL,
        filter_datasets=FILTER_DATASETS,
        protocol_base=cfg["protocol_base"],
        tread_kwargs=tread_kwargs,
    )
    n_rois = trials_df["dff_stim"].iloc[0].shape[1]
    print(f"{session} | {target}: {len(trials_df)} trials, {n_rois} ROIs", flush=True)

    fits = [
        dict(
            model=model,
            choose_trials=choose_trials,
            k_folds=k_folds,
            target=out_dir
            / fit_filename(target, method, model, choose_trials, k_folds),
        )
        for model, choose_trials, k_folds in MODEL_CONFIGS
    ]
    n_tasks = fit_rsof_array.prepare_work_dir(
        work_dir, trials_df, fits, fit_params, roi_batch_size=roi_batch_size
    )
    record = dict(
        session=session,
        target=target,
        method=method,
        protocol_base=cfg["protocol_base"],
        column_suffix=cfg["column_suffix"],
        filter_datasets=FILTER_DATASETS,
        fit_params=fit_params,
        model_configs=MODEL_CONFIGS,
        n_rois=int(n_rois),
        n_trials=int(len(trials_df)),
        n_frames=int(np.sum([len(rs) for rs in trials_df["RS_stim"]])),
        roi_batch_size=roi_batch_size,
        n_tasks=n_tasks,
        frame_index_first=frame_index_offsets(session, cfg["protocol_base"]),
        production_neurons_df=str(neurons_ds.path_full),
        cottage_analysis=code_version(cottage_analysis),
        work_dir=str(work_dir),
        prepared=pd.Timestamp.now().isoformat(timespec="seconds"),
    )
    if not prepare_only:
        array_job, merge_job = fit_rsof_array.submit_array_and_merge(
            work_dir,
            n_tasks,
            mem=mem,
            time_limit=time_limit,
            max_concurrent=max_concurrent,
            cleanup=False,
            job_name=f"rsof_{session}_{target}",
        )
        record.update(array_job=array_job, merge_job=merge_job)
    (out_dir / f"rsof_batched_{target}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("target", choices=["sphere", "treadmill", "treadmill_frames"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--method", default="plateau", choices=["plateau", "model"])
    p.add_argument("--roi-batch-size", type=int, default=25)
    p.add_argument("--time", default="3:00:00")
    p.add_argument("--mem", default="4G")
    p.add_argument("--max-concurrent", type=int, default=200)
    p.add_argument("--prepare-only", action="store_true")
    a = p.parse_args()
    main(
        a.session,
        a.target,
        a.out_dir,
        a.work_root,
        method=a.method,
        roi_batch_size=a.roi_batch_size,
        time_limit=a.time,
        mem=a.mem,
        max_concurrent=a.max_concurrent,
        prepare_only=a.prepare_only,
    )
