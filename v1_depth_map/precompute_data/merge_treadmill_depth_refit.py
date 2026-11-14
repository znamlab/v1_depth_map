"""Merge the fixed-seed treadmill depth refit into the production neurons_df.pickle files.

The refit (`refit_treadmill_depth.py`, NEMO job 63283704) wrote
RESULTS/SESSION/treadmill_depth_fit{_treadmill,_treadmill_model}.pickle and never touched
neurons_df. This script copies the refit columns into SESSION's neurons_df.pickle:

- plateau variant: the `_treadmill` depth columns and their `_treadmill_plateau` twins;
- model variant: the `_treadmill_model` depth columns.

Only the columns the refit actually recomputed are merged (find_depth_neurons, then the
all-trials, odd-trials (`_crossval`) and 5-fold fits). The `_running` / `_notrunning`
columns in some refit files were carried over unchanged from the input neurons_df and are
skipped, so they keep their old (old-seed) values.

Each neurons_df is copied byte-for-byte to `neurons_df.pickle.BACKUP_SUFFIX` first
(the script refuses to run if that backup exists), written atomically, then reloaded:
merged columns must equal the refit and all other columns must be unchanged.

Usage:
    python merge_treadmill_depth_refit.py --processed DIR --results DIR [--dry-run]
DIR for --processed is the project folder holding MOUSE/SESSION/neurons_df.pickle.
"""

import argparse
import json
import os
import pickle
import shutil
import socket
import time
from pathlib import Path

import pandas as pd

VARIANT_FILES = ("treadmill_depth_fit_treadmill", "treadmill_depth_fit_treadmill_model")
SKIP_KEYS = ("_running_", "_notrunning_")  # not refit, carried over from the input
BACKUP_SUFFIX = "pre_depth_seed_fix_backup"


def same(a, b):
    """Columns are equal (works for object columns holding arrays, and NaNs)."""
    return pickle.dumps(a.reset_index(drop=True)) == pickle.dumps(
        b.reset_index(drop=True)
    )


def merge_session(session, processed, results, dry_run):
    mouse, sess = session.split("_", 1)
    path = processed / mouse / sess / "neurons_df.pickle"
    backup = path.with_name(f"{path.name}.{BACKUP_SUFFIX}")
    ndf = pd.read_pickle(path)
    old = ndf.copy()
    merged = {}
    for stem in VARIANT_FILES:
        refit_path = results / session / f"{stem}.pickle"
        refit = pd.read_pickle(refit_path)
        assert (
            refit["roi"]
            .reset_index(drop=True)
            .equals(ndf["roi"].reset_index(drop=True))
        ), f"{session} {stem}: ROI order differs"
        cols = [
            c
            for c in refit.columns
            if c != "roi" and not any(k in c for k in SKIP_KEYS)
        ]
        missing = [c for c in cols if c not in ndf.columns]
        assert not missing, f"{session} {stem}: columns not in neurons_df: {missing}"
        n_changed = 0
        for c in cols:
            n_changed += not same(ndf[c], refit[c])
            ndf[c] = refit[c].values
        merged[stem] = dict(
            file=str(refit_path), columns=cols, n_columns_changed=n_changed
        )
        print(f"{session} {stem}: {len(cols)} columns, {n_changed} changed")
    all_merged = [c for m in merged.values() for c in m["columns"]]
    log = dict(
        session=session,
        neurons_df=str(path),
        backup=str(backup),
        host=socket.gethostname(),
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        dry_run=dry_run,
        merged=merged,
        n_rois=len(ndf),
    )
    if dry_run:
        return log
    if backup.exists():
        raise FileExistsError(f"{backup} exists: not overwriting")
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp_merge")
    ndf.to_pickle(tmp)
    os.replace(tmp, path)
    # verify from disk
    new = pd.read_pickle(path)
    assert list(new.columns) == list(old.columns), "column set changed"
    for c in new.columns:
        ref = ndf[c] if c in all_merged else old[c]
        assert same(new[c], ref), f"{session}: column {c} differs after write"
    print(f"{session}: written and verified, backup {backup.name}")
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sessions = sorted(p.name for p in args.results.iterdir() if p.is_dir())
    logs = [
        merge_session(s, args.processed, args.results, args.dry_run) for s in sessions
    ]
    if not args.dry_run:
        out = (
            args.results.parent / f"merge_log_{socket.gethostname().split('.')[0]}.json"
        )
        out.write_text(json.dumps(logs, indent=2))
        print(f"log: {out}")


if __name__ == "__main__":
    main()
