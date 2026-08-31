"""Make `_treadmill` unambiguously mean "plateau" in every treadmill session's neurons_df.

Background
----------
`cottage_analysis.analysis.treadmill.sync_all_recordings` flipped its default onset-detection
`method` from "model" to "plateau" in commit c4ea1cd (2026-08-17), and `analysis_pipeline.py`
never passes `method` explicitly. The change therefore left *no trace* in any filename or
column name, so `_treadmill` currently means:

  * "model"   in PZAG16.3b_S20250401, PZAG17.3a_S20250402, PZAH17.1e_S20250403 (run 2026-08-14)
  * "plateau" in PZAG16.3c_S20250401                                          (re-run 2026-08-22/23)

This script settles the convention below and rewrites the older sessions to match. Depth tuning
always reduces each trial to its trial-mean dF/F, so it has no per-frame/trial-average axis --
only the onset method distinguishes depth fits. RS/OF has both axes.

    suffix                              depth            RS/OF
    _treadmill                          plateau          per-frame,     plateau
    _treadmill_plateau                  plateau (twin)   --
    _treadmill_model                    model            per-frame,     model
    _treadmill_trial_average_plateau    --               trial-average, plateau  <- figure default
    _treadmill_trial_average_model      --               trial-average, model

Invariant: no treadmill family may mix onset methods, and a suffix means the same thing in
every session.

Also collapses the duplicate `rsof_minSigma_*_x` / `*_y` columns. These are pandas merge
collisions (the k1 and k5 fit pickles both carry `rsof_minSigma_*` under the same name, so
`merge_fit_dataframes`' `reduce(pd.merge, ...)` disambiguates them). Each `_x`/`_y` pair is
byte-identical, so one copy is kept under the un-suffixed base name. They are NOT dropped:
`figure_utils/treadmill.py:654` reads `rsof_minSigma_closedloop_g2d{ta}` to recover the fit's
`min_sigma`, which feeds the ellipse geometry (eccentricity / semimajor / semiminor).

Local-only, idempotent, and backs up `neurons_df.pickle` before writing. Run --dry-run first,
then --apply, then --check.

Usage
-----
    python migrate_treadmill_columns.py --dry-run
    python migrate_treadmill_columns.py --apply
    python migrate_treadmill_columns.py --check
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

PROJECT = "colasa_3d-vision_revisions"

SESSIONS = [
    "PZAG16.3c_S20250401",
    "PZAG16.3b_S20250401",
    "PZAG17.3a_S20250402",
    "PZAH17.1e_S20250403",
]

BACKUP_SUFFIX = ".pre_treadmill_migration_backup"

# Longest first: `_treadmill` is a prefix of every other family, so ordering decides matching.
FAMILIES = [
    "_treadmill_trial_average_plateau",
    "_treadmill_trial_average_model",
    "_treadmill_trial_average",
    "_treadmill_plateau",
    "_treadmill_model",
    "_treadmill",
]

# Substrings that mark a column as belonging to the 1D depth-tuning fit rather than RS/OF.
DEPTH_KEYS = (
    "preferred_depth",
    "depth_tuning",
    "best_depth",
    "is_depth_neuron",
    "depth_neuron_anova",
)

# Families that must exist, and must mean the same thing, in every session once migrated.
EXPECTED_FAMILIES = {
    "_treadmill",
    "_treadmill_plateau",
    "_treadmill_model",
    "_treadmill_trial_average_plateau",
    "_treadmill_trial_average_model",
}


def assert_local_only():
    """Refuse to touch the shared nemo tree -- this script rewrites neurons_df in place."""
    processed_root = str(flz.get_data_root("processed", project=PROJECT))
    if "nemo" in processed_root.lower():
        raise RuntimeError(
            f"Refusing to run: processed root is on nemo ({processed_root})."
        )
    if not processed_root.startswith("/Volumes/BlackPasspo"):
        raise RuntimeError(
            f"Refusing to run: processed root {processed_root} is not /Volumes/BlackPasspo."
        )
    print(f"processed root: {processed_root}")
    return processed_root


def family_of(col):
    """Return the treadmill suffix family of `col`, or None if it carries none."""
    for fam in FAMILIES:
        if col.endswith(fam):
            return fam
    return None


def is_depth(col):
    return any(key in col for key in DEPTH_KEYS)


def session_path(session_name, flexilims_session):
    mouse, sess = session_name.split("_")
    root = Path(flz.get_data_root("processed", project=PROJECT))
    return root / PROJECT / mouse / sess / "neurons_df.pickle"


def summarise(df):
    """Count columns per (family, depth|rsof), for reporting and for the checker."""
    counts = {}
    for col in df.columns:
        fam = family_of(col)
        if fam is None:
            continue
        key = (fam, "depth" if is_depth(col) else "rsof")
        counts[key] = counts.get(key, 0) + 1
    return counts


def print_summary(label, df):
    print(f"  {label}: {len(df.columns)} columns")
    for (fam, kind), n in sorted(summarise(df).items()):
        print(f"      {n:5d}  {kind:5s}  {fam}")


# --------------------------------------------------------------------------------------
# Step 1: collapse the pandas merge-collision duplicates
# --------------------------------------------------------------------------------------


def collapse_xy_duplicates(df, report):
    """Collapse identical `<base>_x` / `<base>_y` pairs down to `<base>`.

    Refuses to collapse a pair whose two members differ -- that would mean the collision hid
    two genuinely different values and a human needs to look.
    """
    xy = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    bases = sorted({c[:-2] for c in xy})

    for base in bases:
        cx, cy = f"{base}_x", f"{base}_y"
        has_x, has_y = cx in df.columns, cy in df.columns

        if has_x and has_y:
            if not df[cx].equals(df[cy]):
                raise RuntimeError(
                    f"{cx} and {cy} differ -- refusing to collapse. Inspect manually."
                )
            if base in df.columns and not df[base].equals(df[cx]):
                raise RuntimeError(
                    f"{base} already exists and differs from {cx} -- refusing to collapse."
                )
            df[base] = df[cx]
            df.drop(columns=[cx, cy], inplace=True)
            report.append(f"collapse  {cx} + {cy}  ->  {base}")
        else:
            # A lone _x or _y: promote it to the base name if that is free.
            lone = cx if has_x else cy
            if base in df.columns:
                df.drop(columns=[lone], inplace=True)
                report.append(f"drop      {lone}  ({base} already present)")
            else:
                df.rename(columns={lone: base}, inplace=True)
                report.append(f"rename    {lone}  ->  {base}")
    return df


# --------------------------------------------------------------------------------------
# Step 2: family renames for the sessions whose `_treadmill` means "model"
# --------------------------------------------------------------------------------------


def treadmill_is_model(df):
    """True if this session's `_treadmill` columns came from the "model" onset method.

    Decided structurally rather than numerically: only a pre-c4ea1cd session carries a
    *RS/OF* `_treadmill_plateau` family alongside `_treadmill`. A post-c4ea1cd session has
    its plateau RS/OF results under the bare `_treadmill` name already.
    """
    return any(
        family_of(c) == "_treadmill_plateau" and not is_depth(c) for c in df.columns
    )


def migrate_model_session(df, report):
    """Rewrite one pre-c4ea1cd session so `_treadmill` holds the plateau results."""
    # 2a. Depth columns wrongly tagged with an RS/OF trial-average basis. Depth has no such
    #     axis; these are byte-identical duplicates of the corresponding `_treadmill[_plateau]`
    #     depth columns, dragged in when merge_fit_dataframes merged the RS/OF fits.
    stale_depth = [
        c
        for c in df.columns
        if is_depth(c)
        and family_of(c)
        in ("_treadmill_trial_average", "_treadmill_trial_average_plateau")
    ]
    for col in stale_depth:
        fam = family_of(col)
        twin = col[: -len(fam)] + (
            "_treadmill_plateau" if "plateau" in fam else "_treadmill"
        )
        if twin in df.columns and not df[col].equals(df[twin]):
            report.append(
                f"WARN      {col} differs from {twin}; dropping anyway (depth has no TA axis)"
            )
    df.drop(columns=stale_depth, inplace=True)
    report.append(
        f"drop      {len(stale_depth)} depth columns tagged _treadmill_trial_average*"
    )

    # 2b. Everything currently under `_treadmill` is model-method -> tag it explicitly.
    #     Covers both depth and RS/OF.
    to_model = [c for c in df.columns if family_of(c) == "_treadmill"]
    df.rename(
        columns={c: c[: -len("_treadmill")] + "_treadmill_model" for c in to_model},
        inplace=True,
    )
    report.append(f"rename    {len(to_model)} columns  _treadmill -> _treadmill_model")

    # 2c. Trial-average RS/OF under the untagged name is likewise model-method.
    to_ta_model = [c for c in df.columns if family_of(c) == "_treadmill_trial_average"]
    df.rename(
        columns={
            c: c[: -len("_treadmill_trial_average")] + "_treadmill_trial_average_model"
            for c in to_ta_model
        },
        inplace=True,
    )
    report.append(
        f"rename    {len(to_ta_model)} columns  _treadmill_trial_average -> _treadmill_trial_average_model"
    )

    # 2d. Promote the per-frame plateau RS/OF results into the now-free `_treadmill` name,
    #     matching what PZAG16.3c already has there.
    rsof_plateau = [
        c
        for c in df.columns
        if family_of(c) == "_treadmill_plateau" and not is_depth(c)
    ]
    df.rename(
        columns={
            c: c[: -len("_treadmill_plateau")] + "_treadmill" for c in rsof_plateau
        },
        inplace=True,
    )
    report.append(
        f"rename    {len(rsof_plateau)} RS/OF columns  _treadmill_plateau -> _treadmill"
    )

    # 2e. Depth: keep `_treadmill_plateau` as the explicit twin AND mirror it onto `_treadmill`,
    #     so bare `_treadmill` means plateau for depth too. The plateau depth fit lacks the
    #     _running/_notrunning splits the model fit has; nothing reads those under a treadmill
    #     suffix, and they survive under `_treadmill_model`.
    depth_plateau = [
        c for c in df.columns if family_of(c) == "_treadmill_plateau" and is_depth(c)
    ]
    for col in depth_plateau:
        target = col[: -len("_treadmill_plateau")] + "_treadmill"
        df[target] = df[col]
    report.append(
        f"copy      {len(depth_plateau)} depth columns  _treadmill_plateau -> _treadmill"
    )
    return df


def migrate_plateau_session(df, report):
    """PZAG16.3c is already correct -- its `_treadmill` is plateau. Nothing to rename.

    It has no `_treadmill_model` / `_treadmill_trial_average_model` families yet; those arrive
    from `precompute_data/fit_revision_treadmill.py --merge` once the model fits land.
    """
    report.append("no family renames needed -- _treadmill is already plateau")
    return df


# --------------------------------------------------------------------------------------
# Checker
# --------------------------------------------------------------------------------------


def agreement(df, col_a, col_b):
    """Fraction of ROIs where two preferred-depth columns agree within 1%."""
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    m = a.notna() & b.notna() & (a > 0) & (b > 0)
    if not m.any():
        return np.nan, 0
    lr = np.abs(np.log(a[m] / b[m]))
    return float((lr < 0.01).mean()), int(m.sum())


def check_session(session_name, path):
    """Assert the post-migration invariants. Returns a list of failure strings."""
    df = pd.read_pickle(path)
    failures = []

    leftover = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if leftover:
        failures.append(f"{len(leftover)} leftover _x/_y columns, e.g. {leftover[:3]}")

    counts = summarise(df)
    fams = {fam for fam, _ in counts}
    unexpected = fams - EXPECTED_FAMILIES
    if unexpected:
        failures.append(f"unexpected families: {sorted(unexpected)}")

    depth_ta = [(f, k) for (f, k) in counts if k == "depth" and "trial_average" in f]
    if depth_ta:
        failures.append(f"depth columns carrying a trial_average tag: {depth_ta}")

    rsof_treadmill = counts.get(("_treadmill", "rsof"), 0)
    depth_treadmill = counts.get(("_treadmill", "depth"), 0)
    if rsof_treadmill == 0:
        failures.append("no RS/OF columns under _treadmill")
    if depth_treadmill == 0:
        failures.append("no depth columns under _treadmill")

    # Column *presence* is not enough. A promoted family can be structurally complete and
    # still be empty: the treadmill plateau depth fits carried `preferred_depth` but no
    # cross-validated Spearman statistics, so `is_depth_neuron_treadmill` came out
    # all-False and silently emptied every depth-cell selection downstream. Check that the
    # columns the figures actually select on are populated and non-degenerate.
    for col, kind in [
        ("is_depth_neuron_treadmill", "flag"),
        ("depth_tuning_test_spearmanr_rval_closedloop_treadmill", "stat"),
        ("depth_tuning_test_spearmanr_pval_closedloop_treadmill", "stat"),
    ]:
        if col not in df.columns:
            failures.append(f"missing {col}")
            continue
        if kind == "stat":
            n = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            if n == 0:
                failures.append(
                    f"{col} is entirely empty -- the fit wrote no test stats"
                )
        else:
            n_true = int(df[col].fillna(False).astype(bool).sum())
            if n_true == 0:
                failures.append(
                    f"{col} has no True values -- depth-cell selections will be empty"
                )
            else:
                print(f"      {col}: {n_true} depth neurons")

    # The check that would have caught the original bug: bare _treadmill depth must agree
    # with the explicit plateau twin.
    a_col = "preferred_depth_closedloop_crossval_treadmill"
    p_col = "preferred_depth_closedloop_crossval_treadmill_plateau"
    if a_col in df.columns and p_col in df.columns:
        frac, n = agreement(df, a_col, p_col)
        if not (frac >= 0.95):
            failures.append(
                f"_treadmill depth disagrees with _treadmill_plateau ({frac:.1%} of {n} ROIs)"
            )
        else:
            print(
                f"      _treadmill vs _treadmill_plateau depth: {frac:.1%} of {n} ROIs agree"
            )
    else:
        failures.append(f"missing {a_col} or {p_col}")

    # And the model twin, where present, must be genuinely different -- proving the rename
    # tagged real model results rather than duplicating plateau.
    m_col = "preferred_depth_closedloop_crossval_treadmill_model"
    if m_col in df.columns:
        frac, n = agreement(df, m_col, p_col)
        if frac >= 0.95:
            failures.append(
                f"_treadmill_model depth is ~identical to plateau ({frac:.1%}) -- suspect copy"
            )
        else:
            print(
                f"      _treadmill_model vs plateau depth: {frac:.1%} of {n} ROIs agree (expect low)"
            )

    print(f"  {session_name}: {len(df.columns)} columns")
    for (fam, kind), n in sorted(counts.items()):
        print(f"      {n:5d}  {kind:5s}  {fam}")
    return failures


# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="Report planned changes only."
    )
    mode.add_argument("--apply", action="store_true", help="Rewrite neurons_df.pickle.")
    mode.add_argument(
        "--check", action="store_true", help="Verify post-migration invariants."
    )
    parser.add_argument("--sessions", nargs="+", default=SESSIONS)
    args = parser.parse_args()

    assert_local_only()
    flm = flz.get_flexilims_session(project_id=PROJECT)

    all_failures = {}
    for session_name in args.sessions:
        path = session_path(session_name, flm)
        print(f"\n=== {session_name}")
        if not path.exists():
            print(f"  SKIP: {path} not found")
            continue

        if args.check:
            failures = check_session(session_name, path)
            if failures:
                all_failures[session_name] = failures
            continue

        df = pd.read_pickle(path)
        print_summary("before", df)

        report = []
        df = collapse_xy_duplicates(df, report)
        if treadmill_is_model(df):
            print("  detected: _treadmill == model  (pre-c4ea1cd session)")
            df = migrate_model_session(df, report)
        else:
            print("  detected: _treadmill == plateau (post-c4ea1cd session)")
            df = migrate_plateau_session(df, report)

        print("  changes:")
        for line in report:
            print(f"      {line}")
        print_summary("after ", df)

        if args.apply:
            backup = Path(str(path) + BACKUP_SUFFIX)
            if backup.exists():
                print(f"  backup already exists, not overwriting: {backup.name}")
            else:
                shutil.copy2(path, backup)
                print(f"  backed up -> {backup.name}")
            df.to_pickle(path)
            print(f"  wrote {path.name}")
        else:
            print("  [dry-run] not written")

    if args.check:
        print("\n=== check summary")
        if all_failures:
            for session_name, failures in all_failures.items():
                print(f"  FAIL {session_name}")
                for f in failures:
                    print(f"       - {f}")
            raise SystemExit(1)
        print("  all sessions pass")


if __name__ == "__main__":
    main()
