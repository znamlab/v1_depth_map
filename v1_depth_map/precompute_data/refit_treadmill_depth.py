"""Refit the treadmill 1D depth tuning of one session, then merge it into neurons_df.

Fit step: for one onset-detection variant from `VARIANTS`, run find_depth_neurons, then
fit_preferred_depth on all trials, on odd trials (`_crossval`) and with 5-fold
cross-validation, with `max_rs2motor_diff=0.3`. The `_running` / `_notrunning` fits are
not redone. The session's `neurons_df.pickle` is only read (for ROI alignment); the depth
columns of the variant (and their mirror, e.g. `_treadmill_plateau` beside `_treadmill`)
are saved, with `roi`, to OUT_DIR/SESSION/treadmill_depth_fit{suffix}.pickle, with
provenance in a .json.

Needs cottage_analysis f6d95de or later: `best_depth` is NaN-aware and
`fit_preferred_depth` restarts from depths spread over the tested range instead of
reusing the best-depth seed for every restart (the old seed left most treadmill plateau
fits flat, preferred depth at a fit bound, R^2 ~ 0). The random restarts are not seeded,
so a rerun matches the shipped fits only up to optimiser noise.

Merge step (`--merge`): copies the refit columns into the session's `neurons_df.pickle`
(located through flexiznam, as in the fit step), skipping the `_running_` /
`_notrunning_` columns that the refit carried over unchanged. See `merge_into_neurons_df`
for the backup, atomic write and verification.

Usage:
    python refit_treadmill_depth.py SESSION VARIANT --out-dir DIR [--overwrite]
    python refit_treadmill_depth.py SESSION VARIANT --out-dir DIR --merge [--dry-run]
        [--backup-suffix SUFFIX] [--overwrite-backup]
VARIANT is a name from `VARIANTS` (plateau_single_frames, model_single_frames).
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

import cottage_analysis
import v1_depth_map
from cottage_analysis.analysis import common_utils, find_depth_neurons
from cottage_analysis.pipelines import pipeline_utils

PROJECT = "colasa_3d-vision_revisions"
PHOTODIODE_PROTOCOL = 5
MAX_RS2MOTOR_DIFF = 0.3

# Substrings marking a column as 1D depth-tuning output rather than RS/OF.
DEPTH_COLUMN_KEYS = (
    "preferred_depth",
    "depth_tuning",
    "best_depth",
    "is_depth_neuron",
    "depth_neuron_anova",
)

# Treadmill depth variants. The onset method is always explicit in the output name: bare
# `_treadmill` IS the plateau family, and `_treadmill_plateau` is written alongside it as
# an identical self-documenting twin (`mirror_suffix`). See "Column naming" in
# fit_revision_treadmill.py for the full convention.
VARIANTS = [
    {
        "name": "plateau_single_frames",
        "method": "plateau",
        "suffix": "_treadmill",
        "mirror_suffix": "_treadmill_plateau",
    },
    {
        "name": "model_single_frames",
        "method": "model",
        "suffix": "_treadmill_model",
        "mirror_suffix": None,
    },
]

# carried over unchanged from the input neurons_df by the treadmill refit: never merged
TREADMILL_SKIP_KEYS = ("_running_", "_notrunning_")


def code_version(module):
    """Git commit (and whether the tree had local changes) of an installed package."""
    root = Path(module.__file__).resolve().parents[1]
    marker = root / "CODE_VERSION.txt"  # written when the code is copied to the run dir
    if marker.exists():
        return marker.read_text().strip()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet"], check=False
        ).returncode
        return commit + (" (local changes)" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_neurons_ds(session, flexilims_session=None):
    """The session's neurons_df dataset (its `path_full` is the neurons_df.pickle)."""
    if flexilims_session is None:
        flexilims_session = flz.get_flexilims_session(project_id=PROJECT)
    return flz.get_datasets(
        origin_name=session,
        dataset_type="neurons_df",
        flexilims_session=flexilims_session,
        allow_multiple=False,
    )


# ---------------------------------------------------------------------------------------
# merge into neurons_df (shared with refit_closedloop_depth.py)


def _is_missing(x):
    return x is None or x is pd.NA or x is pd.NaT or (isinstance(x, float) and x != x)


def _cell_equal(x, y):
    if _is_missing(x) or _is_missing(y):
        return _is_missing(x) and _is_missing(y)
    if isinstance(x, (np.ndarray, list, tuple)) or isinstance(
        y, (np.ndarray, list, tuple)
    ):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return False
        try:
            return bool(np.array_equal(x, y, equal_nan=True))
        except TypeError:  # non-numeric arrays
            return bool(np.array_equal(x, y))
    if isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating)):
        return x == y or (np.isnan(x) and np.isnan(y))
    eq = x == y
    return bool(np.all(eq)) if isinstance(eq, np.ndarray) else bool(eq)


def columns_equal(a, b):
    """Same dtype and values, by position; NaN-aware, array-valued cells elementwise."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    if len(a) != len(b) or a.dtype != b.dtype:
        return False
    if a.dtype != object:
        return a.equals(b)
    return all(_cell_equal(x, y) for x, y in zip(a, b))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_pickle(df, path):
    df.to_pickle(path)


def merge_into_neurons_df(
    neurons_df_path,
    result_path,
    skip_keys=(),
    backup_suffix=None,
    dry_run=False,
    overwrite_backup=False,
    log_path=None,
):
    """Copy the columns of a refit result into a neurons_df.pickle.

    Rows are matched by `roi` (same ROIs in the same order, else AssertionError). Every
    column of `result_path` except `roi` and those containing one of `skip_keys` is
    written over the column of the same name; columns absent from neurons_df are added
    at the end. Unless `dry_run`:

    1. neurons_df.pickle is copied byte-for-byte to `neurons_df.pickle.{backup_suffix}`
       (refuses if that file exists, unless `overwrite_backup`);
    2. the merged frame is written to a temporary file in the same folder and moved over
       neurons_df.pickle with os.replace;
    3. the file is reloaded: merged columns must equal the refit, every other column must
       equal the old one (NaN-aware, array-valued cells elementwise), and the old columns
       must keep their order. On failure, the backup is restored and AssertionError raised;
    4. a log entry (columns merged/changed/added, sha256 of the files, code versions) is
       appended to `log_path` (default: RESULT.merge_log.json next to the result).

    Returns the log entry (also printed). A dry run reads both files and writes nothing.
    """
    neurons_df_path, result_path = Path(neurons_df_path), Path(result_path)
    if backup_suffix is None:
        backup_suffix = f"pre_{result_path.stem}_merge_{time.strftime('%Y%m%d')}"
    backup = neurons_df_path.with_name(f"{neurons_df_path.name}.{backup_suffix}")
    if log_path is None:
        log_path = result_path.with_suffix(".merge_log.json")
    log_path = Path(log_path)

    old = pd.read_pickle(neurons_df_path)
    result = pd.read_pickle(result_path)
    assert (
        result["roi"].reset_index(drop=True).equals(old["roi"].reset_index(drop=True))
    ), f"ROIs of {result_path} differ from {neurons_df_path} (set or order)"
    cols = [
        c for c in result.columns if c != "roi" and not any(k in c for k in skip_keys)
    ]
    skipped = [c for c in result.columns if c != "roi" and c not in cols]
    added = [c for c in cols if c not in old.columns]
    changed = [
        c for c in cols if c in old.columns and not columns_equal(old[c], result[c])
    ]
    new = old.copy()
    for c in cols:
        new[c] = result[c].values

    fit_json = result_path.with_suffix(".json")
    log = dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=socket.gethostname(),
        dry_run=dry_run,
        neurons_df=str(neurons_df_path),
        neurons_df_sha256_before=sha256(neurons_df_path),
        result=str(result_path),
        result_sha256=sha256(result_path),
        fit_provenance=json.loads(fit_json.read_text()) if fit_json.exists() else None,
        backup=None if dry_run else str(backup),
        n_rois=len(old),
        columns_merged=cols,
        columns_changed=changed,
        columns_added=added,
        columns_skipped=skipped,
        v1_depth_map=code_version(v1_depth_map),
    )
    print(
        f"{neurons_df_path}: {len(cols)} columns from {result_path.name}, "
        f"{len(changed)} changed, {len(added)} added {added}, {len(skipped)} skipped",
        flush=True,
    )
    if dry_run:
        print("dry run: nothing written")
        return log

    if backup.exists() and not overwrite_backup:
        raise FileExistsError(
            f"{backup} exists: not overwriting (pass --overwrite-backup or choose "
            "another --backup-suffix)"
        )
    shutil.copy2(neurons_df_path, backup)
    tmp = neurons_df_path.with_name(neurons_df_path.name + ".tmp_merge")
    try:
        _write_pickle(new, tmp)
        os.replace(tmp, neurons_df_path)
    finally:
        if tmp.exists():
            tmp.unlink()

    written = pd.read_pickle(neurons_df_path)
    errors = []
    if list(written.columns) != list(old.columns) + added:
        errors.append("column order changed")
    for c in written.columns:
        ref = result[c] if c in cols else old.get(c)
        if ref is None or not columns_equal(written[c], ref):
            errors.append(c)
    if errors:
        shutil.copy2(backup, tmp)
        os.replace(tmp, neurons_df_path)
        raise AssertionError(
            f"{neurons_df_path}: verification failed for {errors[:10]}; restored the "
            f"backup {backup.name}"
        )
    log["neurons_df_sha256_after"] = sha256(neurons_df_path)
    print(f"written and verified, backup {backup.name}", flush=True)

    logs = json.loads(log_path.read_text()) if log_path.exists() else []
    log_path.write_text(json.dumps(logs + [log], indent=2))
    print(f"log: {log_path}")
    return log


def add_merge_arguments(parser):
    """CLI options of the merge step, shared by the refit scripts."""
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge the result in OUT_DIR/SESSION into the session's neurons_df.pickle "
        "instead of fitting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --merge: report what would change, write nothing",
    )
    parser.add_argument(
        "--backup-suffix",
        default=None,
        help="With --merge: backup is neurons_df.pickle.SUFFIX (default "
        "pre_RESULT_merge_YYYYMMDD)",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="With --merge: replace an existing backup of the same name",
    )


# ---------------------------------------------------------------------------------------
# fit


def result_path(out_dir, session, variant_name):
    suffix = {v["name"]: v for v in VARIANTS}[variant_name]["suffix"]
    return Path(out_dir) / session / f"treadmill_depth_fit{suffix}.pickle"


def main(session, variant_name, out_dir, overwrite=False):
    var = {v["name"]: v for v in VARIANTS}[variant_name]
    suffix, mirror = var["suffix"], var.get("mirror_suffix")
    out = result_path(out_dir, session, variant_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} exists: not overwriting (pass --overwrite to replace)"
        )

    neurons_ds = get_neurons_ds(session)
    neurons_df = pd.read_pickle(neurons_ds.path_full)  # read only

    t0 = time.time()
    _, _, _, trials_df = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets={"anatomical_only": 3, "annotated": True},
        protocol_base="SpheresTubeMotor",
        recording_type="two_photon",
        tread_kwargs={"method": var["method"]},
    )
    trials_df = trials_df[~trials_df.recording_name.str.contains("multidepth")]
    print(f"{session}: loaded trials in {time.time() - t0:.0f} s", flush=True)

    ndf, nds = find_depth_neurons.find_depth_neurons(
        trials_df=trials_df,
        neurons_ds=neurons_ds,
        neurons_df=neurons_df.copy(),
        rs_thr=None,
        alpha=0.05,
        special_sfx=suffix,
        max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
    )
    depth_min = np.round(trials_df.depth.min(), 4)
    depth_max = np.ceil(trials_df.depth.max())
    for k_folds, choose_trials in [(1, None), (1, "odd"), (5, None)]:
        t1 = time.time()
        ndf, nds = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df,
            neurons_df=ndf,
            neurons_ds=nds,
            depth_min=depth_min,
            depth_max=depth_max,
            k_folds=k_folds,
            choose_trials=choose_trials,
            special_sfx=suffix,
            max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
        )
        print(
            f"{session}: fit k_folds={k_folds} trials={choose_trials} "
            f"in {time.time() - t1:.0f} s",
            flush=True,
        )
    common_utils.add_one_sided_spearman_significance(
        ndf,
        rval_col=f"depth_tuning_test_spearmanr_rval_closedloop{suffix}",
        pval_col=f"depth_tuning_test_spearmanr_pval_closedloop{suffix}",
        out_col=f"is_depth_neuron{suffix}",
    )

    # depth columns of this variant only (endswith is exact: "_treadmill" does not match
    # "_treadmill_model"), plus their mirror twin
    cols = [
        c
        for c in ndf.columns
        if c.endswith(suffix) and any(key in c for key in DEPTH_COLUMN_KEYS)
    ]
    result = ndf[["roi"] + cols].copy()
    if mirror:
        for c in cols:
            result[c[: -len(suffix)] + mirror] = result[c]
    assert result.roi.equals(neurons_df.roi), "ROI order changed"
    result.to_pickle(out)

    old_sel = neurons_df.get(f"is_depth_neuron{suffix}")
    provenance = dict(
        session=session,
        variant=variant_name,
        suffix=suffix,
        mirror_suffix=mirror,
        method=var["method"],
        max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
        depth_min=float(depth_min),
        depth_max=float(depth_max),
        n_trials=int(len(trials_df)),
        neurons_df=str(neurons_ds.path_full),
        cottage_analysis=code_version(cottage_analysis),
        columns=list(result.columns),
        n_depth_selective_new=int(result[f"is_depth_neuron{suffix}"].sum()),
        n_depth_selective_old=None if old_sel is None else int(old_sel.sum()),
        runtime_s=round(time.time() - t0),
    )
    out.with_suffix(".json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2))


def merge(session, variant_name, out_dir, **kwargs):
    """Merge OUT_DIR/SESSION/treadmill_depth_fit{suffix}.pickle into neurons_df."""
    return merge_into_neurons_df(
        get_neurons_ds(session).path_full,
        result_path(out_dir, session, variant_name),
        skip_keys=TREADMILL_SKIP_KEYS,
        **kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("session")
    parser.add_argument("variant", choices=[v["name"] for v in VARIANTS])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )
    add_merge_arguments(parser)
    args = parser.parse_args()
    if not args.merge and (args.dry_run or args.backup_suffix or args.overwrite_backup):
        parser.error("--dry-run, --backup-suffix and --overwrite-backup need --merge")
    if args.merge:
        merge(
            args.session,
            args.variant,
            args.out_dir,
            backup_suffix=args.backup_suffix,
            dry_run=args.dry_run,
            overwrite_backup=args.overwrite_backup,
        )
    else:
        main(args.session, args.variant, args.out_dir, overwrite=args.overwrite)
