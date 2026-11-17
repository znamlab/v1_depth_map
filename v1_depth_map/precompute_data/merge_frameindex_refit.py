"""Merge the FrameIndex-fix refit of the PZAG22 sessions into their neurons_df.pickle.

The refit run (`refit_rsof_batched.py`, `refit_closedloop_depth.py`,
`refit_treadmill_depth.py`, NEMO run directory `frameindex_refit`) wrote
RESULTS/SESSION/ and never touched the session folders. This script copies its columns
into SESSION's neurons_df.pickle:

- RS/OF fits: per target, the `fit_rs_of_tuning_*` files are renamed and joined exactly as
  `pipeline_utils.merge_fit_dataframes` does (column suffix of
  `fit_revision_treadmill.FIT_TARGETS`, files joined on `roi`, so duplicated metadata
  columns get `_x` / `_y`), then written over the columns of the same name:
  `sphere` -> no suffix (closed loop), `treadmill_frames` -> `_treadmill`,
  `treadmill` -> `_treadmill_trial_average_plateau`;
- closed-loop depth (`closedloop_depth_fit.pickle`): every column, including the
  `_running` / `_notrunning` fits, which this refit redid;
- treadmill depth (`treadmill_depth_fit_treadmill[_model].pickle`): as
  `merge_treadmill_depth_refit.py`, the `_running_` / `_notrunning_` columns are skipped
  (not refit, carried over from the input).

Columns absent from neurons_df are added (listed in the log). Each neurons_df is copied
byte-for-byte to `neurons_df.pickle.pre_frameindex_fix_backup` first (refuses to run if
that backup exists), written atomically, then reloaded and checked: merged columns equal
the refit, all other columns unchanged.

Usage:
    python merge_frameindex_refit.py --processed DIR --results DIR [--dry-run]
"""

import argparse
import json
import os
import pickle
import re
import shutil
import socket
import time
from functools import reduce
from pathlib import Path

import pandas as pd

BACKUP_SUFFIX = "pre_frameindex_fix_backup"
# filename tag -> column suffix (fit_revision_treadmill.target_config(target, "plateau"))
RSOF_TARGETS = {
    "_legacy_refit": "",
    "_treadmill_legacy_plateau": "_treadmill",
    "_treadmill_trial_average_legacy_plateau": "_treadmill_trial_average_plateau",
}
FIT_NAME = re.compile(r"fit_rs_of_tuning_(.+?_k\d)(_.*)\.pickle")
TREADMILL_DEPTH = (
    "treadmill_depth_fit_treadmill",
    "treadmill_depth_fit_treadmill_model",
)
SKIP_KEYS = ("_running_", "_notrunning_")


def same(a, b):
    """Columns are equal (works for object columns holding arrays, and NaNs)."""
    return pickle.dumps(a.reset_index(drop=True)) == pickle.dumps(
        b.reset_index(drop=True)
    )


def rsof_frames(sdir):
    """{column_suffix: merged dataframe} as merge_fit_dataframes would build it."""
    by_tag = {}
    for f in sorted(sdir.glob("fit_rs_of_tuning_*.pickle")):
        tag = FIT_NAME.match(f.name).group(2)
        by_tag.setdefault(tag, []).append(f)
    out = {}
    for tag, files in by_tag.items():
        sfx = RSOF_TARGETS[tag]
        dfs = []
        for f in files:
            df = pd.read_pickle(f)
            df.columns = [c if c == "roi" else f"{c}{sfx}" for c in df.columns]
            dfs.append(df)
        out[sfx] = (
            reduce(lambda x, y: pd.merge(x, y, on="roi", how="inner"), dfs),
            [f.name for f in files],
        )
    return out


def merge_session(session, processed, results, dry_run):
    mouse, sess = session.split("_", 1)
    path = processed / mouse / sess / "neurons_df.pickle"
    backup = path.with_name(f"{path.name}.{BACKUP_SUFFIX}")
    sdir = results / session
    ndf = pd.read_pickle(path)
    old = ndf.copy()
    roi = ndf["roi"].reset_index(drop=True)
    sources = []  # (label, dataframe, columns to merge, files)

    for sfx, (df, files) in rsof_frames(sdir).items():
        sources.append(
            (f"rsof{sfx or '_closedloop'}", df, [c for c in df if c != "roi"], files)
        )
    cl = sdir / "closedloop_depth_fit.pickle"
    if cl.exists():
        df = pd.read_pickle(cl)
        sources.append(
            ("depth_closedloop", df, [c for c in df if c != "roi"], [cl.name])
        )
    for stem in TREADMILL_DEPTH:
        f = sdir / f"{stem}.pickle"
        if f.exists():
            df = pd.read_pickle(f)
            cols = [c for c in df if c != "roi" and not any(k in c for k in SKIP_KEYS)]
            sources.append((stem, df, cols, [f.name]))

    merged, all_merged = {}, []
    for label, df, cols, files in sources:
        assert (
            df["roi"].reset_index(drop=True).equals(roi)
        ), f"{session} {label}: ROI order"
        dup = set(cols) & set(all_merged)
        assert not dup, f"{session} {label}: columns merged twice: {sorted(dup)[:5]}"
        added = [c for c in cols if c not in ndf.columns]
        n_changed = sum(not same(ndf[c], df[c]) for c in cols if c in ndf.columns)
        for c in cols:
            ndf[c] = df[c].values
        all_merged += cols
        merged[label] = dict(
            files=files,
            n_columns=len(cols),
            n_columns_changed=n_changed,
            columns_added=added,
        )
        print(
            f"{session} {label}: {len(cols)} columns, {n_changed} changed, {len(added)} added"
        )

    log = dict(
        session=session,
        neurons_df=str(path),
        backup=str(backup),
        host=socket.gethostname(),
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        dry_run=dry_run,
        n_rois=len(ndf),
        merged=merged,
    )
    if dry_run:
        return log
    if backup.exists():
        raise FileExistsError(f"{backup} exists: not overwriting")
    shutil.copy2(path, backup)
    tmp = path.with_name(path.name + ".tmp_merge")
    ndf.to_pickle(tmp)
    os.replace(tmp, path)
    new = pd.read_pickle(path)
    assert list(new.columns)[: len(old.columns)] == list(
        old.columns
    ), "column order changed"
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
    sessions = sorted(
        p.name for p in args.results.iterdir() if p.is_dir() and "_S" in p.name
    )
    logs = [
        merge_session(s, args.processed, args.results, args.dry_run) for s in sessions
    ]
    tag = "dry_run" if args.dry_run else socket.gethostname().split(".")[0]
    out = args.results.parent / f"merge_log_{tag}.json"
    out.write_text(json.dumps(logs, indent=2))
    print(f"log: {out}")


if __name__ == "__main__":
    main()
