"""Re-run the RS/OF tuning fits for the revision treadmill sessions.

The four `colasa_3d-vision_revisions` sessions that have a `SpheresTubeMotor` recording
are fit up to three ways:

- the **sphere** (closed-loop `SpheresPermTubeReward`) part **frame by frame**, as the
  standard pipeline does, writing `fit_rs_of_tuning_<model>[_crossval]_k<n>.pickle`;
- the **treadmill** (`SpheresTubeMotor`) part on **trial averages**, writing
  `fit_rs_of_tuning_<model>[_crossval]_k<n>_treadmill_trial_average_legacy.pickle`;
- the **treadmill_frames** part, the same `SpheresTubeMotor` recordings but **frame by
  frame** like the sphere target, writing
  `fit_rs_of_tuning_<model>[_crossval]_k<n>_treadmill_legacy.pickle`. This mirrors the
  standard pipeline's own per-frame `SpheresTubeMotor` fit (`analysis_pipeline.py`'s
  `run_rsof_fit` block, `max_rs2motor_diff=0.3`, `file_special_sfx="_treadmill"`) so its
  `method="plateau"` output reproduces the real per-frame `*_treadmill` columns already in
  `neurons_df.pickle`.

Column naming
-------------
Onset-detection method is ALWAYS explicit in the merged column names. `_treadmill` means
plateau everywhere (see `revisions/migrate_treadmill_columns.py`, which settled that
convention across all four sessions), so a `model` fit must never merge into it:

    target            method     merges into
    treadmill         plateau    _treadmill_trial_average_plateau   <- figure default
    treadmill         model      _treadmill_trial_average_model
    treadmill_frames  plateau    _treadmill
    treadmill_frames  model      _treadmill_model

Note the deliberate asymmetry: bare `_treadmill` IS the per-frame plateau family (it is read
at ~240 hardcoded sites and keeps its name), while the trial-average family tags both
methods. The mapping is spelled out in `FIT_TARGETS[...]["column_suffix"]` rather than
derived by appending `_{method}`, so this cannot drift.

All three use the *legacy* ``(log_sigma_x2, log_sigma_y2, theta)`` 2D-Gaussian
parameterisation of the `reviews` branch of cottage_analysis. That is the point of the
`_legacy` filename tag: nemo carries identically-named `..._treadmill_trial_average.pickle`
files whose popts are in the newer Cholesky parameterisation, and the two are NOT
interchangeable. The tag makes them impossible to confuse if anything is ever synced
between the two locations. (The real per-frame `*_treadmill.pickle` family happens to
already be legacy-parameterised for this project - see `treadmill.ipynb` cell 23 - so
`treadmill_frames`'s `_legacy` tag is for naming consistency rather than a parameterisation
guard, but it also keeps the filename from colliding with that real, untagged family.)

**Which storage this writes to is controlled by `--site {local,nemo}`.** With `--site local`
(the default), `~/.flexiznam/config.yml` has an explicit `project_paths` entry mapping
`colasa_3d-vision_revisions` to `/Volumes/BlackPasspo/v1_depth_map/{processed,raw}`, and
`neurons_ds.path_full` resolves through `flz.get_data_root` to that personal copy.
`--site nemo` instead expects (and `assert_site_root` checks) that the same call resolves to
the real shared `/nemo/lab/znamenskiyp/home/shared/projects` tree -- i.e. it is meant to be run
directly on the cluster (where `--use-slurm` needs `sbatch` on `PATH` anyway), writing into the
production session folders. The `_legacy`/`_plateau`-style filename and `column_suffix` tags
are what keep that run from colliding with what is already there, most notably the real,
differently-parameterised `..._treadmill_trial_average.pickle` files nemo already carries (see
the `_legacy` tag note above) -- not a promise that this script never touches nemo.

Typical use::

    # time one config before committing to the full set
    python fit_revision_treadmill.py --sessions PZAG17.3a_S20250402 --only treadmill \\
        --configs gaussian_2d:None:1

    python fit_revision_treadmill.py              # all sessions, all 3 targets, 11 configs
    python fit_revision_treadmill.py --merge      # merge results into neurons_df.pickle

`--method plateau` re-fits a `SpheresTubeMotor` target (`treadmill` or `treadmill_frames`) with
`treadmill.sync_all_recordings`'s trapezoidal-ramp onset detector instead of the default
`"model"` heuristic (the `sphere` target is unaffected, since it never goes through
`treadmill.sync_all_recordings`). Output filenames/columns get a `_plateau` tag so they land
alongside the `"model"` results instead of overwriting them::

    python fit_revision_treadmill.py --only treadmill --method plateau
    python fit_revision_treadmill.py --only treadmill --method plateau --merge

    # frame-by-frame plateau fit, comparable to the real per-frame *_treadmill columns
    python fit_revision_treadmill.py --only treadmill_frames --method plateau
    python fit_revision_treadmill.py --only treadmill_frames --method plateau --merge

Fit bounds (`param_range`)
--------------------------
`param_range` bounds the *centre* of the fitted Gaussian - `x0` = log(RS in m/s), `y0` =
log(OF in deg/s) - and nothing else. It is **per fit target**, not shared, because the
three targets sample very different boxes:

    target            param_range              why
    sphere            DEFAULT_PARAM_RANGE      free running, genuinely spans 0.5-500 cm/s
    treadmill         TREADMILL_PARAM_RANGE    the belt's own grid, padded 1 min-sigma
    treadmill_frames  DEFAULT_PARAM_RANGE      unchanged, so it still reproduces the
                                               production per-frame `*_treadmill` columns

The treadmill grid is 5 belt speeds (3.8125-61 cm/s) x 6 optic flows (1-1024 deg/s), two
orders of magnitude smaller than the default box. Under the default bounds 24-32% of ROIs
per session had their preferred RS or OF pinned on a bound, where the Gaussian is a
monotonic ramp and the "preferred" value is an artefact of the bound. See
`TREADMILL_PARAM_RANGE` for the derivation.

`--check-range` is read-only: it loads each session, reports the empirical trial-averaged
RS/OF box against the target's `param_range`, and stops. **Run it before any refit** - it is
what confirms the protocol-grid constants against the real data::

    python fit_revision_treadmill.py --only treadmill --method plateau --check-range

Because `fit_rs_of_tuning` records `min_sigma` in its output but NOT `param_range`, and
because the g2d refit overwrites `_treadmill_trial_average_plateau` in place, every real run
also writes `param_range_current.json` into the session folder (invisible to
`merge_fit_dataframes`' `*.pickle` glob). Without it there is no way to tell after the fact
which bounds produced a given pickle, so back up the previous pickles alongside their own
record before re-fitting.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.analysis import fit_gaussian_blob, treadmill
from cottage_analysis.pipelines import pipeline_utils

PROJECT = "colasa_3d-vision_revisions"

# The four sessions with a SpheresTubeMotor recording.
SESSIONS = [
    "PZAG16.3b_S20250401",
    "PZAG16.3c_S20250401",
    "PZAG17.3a_S20250402",
    "PZAH17.1e_S20250403",
]

# Kept in sync with cottage_analysis.pipelines.analysis_pipeline (the `to_do` list of the
# run_rsof_fit block). (model, choose_trials, k_folds).
MODEL_CONFIGS = [
    ("gaussian_2d", None, 1),
    ("gaussian_2d", "even", 1),
    ("gaussian_additive", None, 1),
    ("gaussian_OF", None, 1),
    ("gaussian_2d", None, 5),
    ("gaussian_additive", None, 5),
    ("gaussian_OF", None, 5),
    ("gaussian_ratio", None, 1),
    ("gaussian_ratio", None, 5),
    ("gaussian_RS", None, 1),
    ("gaussian_RS", None, 5),
]

# `min_sigma` is added to sigma *squared* inside the Gaussian models
# (`sigma_x_sq = np.exp(log_sigma_x2) + min_sigma`, fit_gaussian_blob.gaussian_2d), so one
# minimum standard deviation in natural-log units is sqrt(min_sigma), not min_sigma.
MIN_SIGMA = 0.25
_LOG_MARGIN = np.sqrt(MIN_SIGMA)  # 0.5 ln-units -> a factor exp(0.5) = 1.6487 each side

# `param_range` bounds only the *centre* of the fitted Gaussian: `x0` = log(RS in m/s) and
# `y0` = log(OF in deg/s) (fit_gaussian_blob.initial_fit_conditions). Amplitude, sigmas and
# offset stay unbounded.
#
# The production range, sized for the free-running sphere protocol (RS 0.5-500 cm/s,
# OF 0.03-3000 deg/s).
DEFAULT_PARAM_RANGE = {"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000}

# The treadmill protocol samples a far smaller box than that, so with the default range a
# large fraction of trial-average fits (24-32% of ROIs per session, measured on the
# 2026-08-07 plateau pickles) ran their preferred RS or OF out to a bound and stopped
# there: the Gaussian degenerates into a monotonic ramp whose "preferred" value is an
# artefact of the bound rather than of the data.
#
# The box is the protocol's own grid, derived from the constants rather than typed out:
# five belt speeds from treadmill.ACTUAL_MOTOR_SPEED (3.8125-61 cm/s) crossed with the
# expected optic flows of treadmill.compute_response_matrix (4**arange(6), 1-1024 deg/s).
# It is padded by one MIN_SIGMA so a cell whose true preference sits on the edge of the
# sampled range can still be fit, without allowing a centre so far outside the stimulus
# that it is unconstrained. `--check-range` verifies this box against the real data.
#
# `max_rs2motor_diff=0.3` already keeps measured RS within 30% of the commanded belt speed
# (a factor of 1.3, inside the 1.6487 pad), so the pad covers animal/motor deviation too.
_TM_RS = np.array(sorted(treadmill.ACTUAL_MOTOR_SPEED.values())) / 100.0  # cm/s -> m/s
_TM_OF = 4.0 ** np.arange(6)  # deg/s
TREADMILL_PARAM_RANGE = {
    "rs_min": float(_TM_RS.min() / np.exp(_LOG_MARGIN)),  # 0.023124 m/s
    "rs_max": float(_TM_RS.max() * np.exp(_LOG_MARGIN)),  # 1.005720 m/s
    "of_min": float(_TM_OF.min() / np.exp(_LOG_MARGIN)),  # 0.606531 deg/s
    "of_max": float(_TM_OF.max() * np.exp(_LOG_MARGIN)),  # 1688.291 deg/s
}

# Shared fit parameters, matching analysis_pipeline's `common_params`. `param_range` is
# NOT here: it is per fit target (see FIT_TARGETS), because the sphere target's
# free-running RS legitimately spans the wide default range.
COMMON_PARAMS = dict(
    rs_thr=0.01,
    niter=10,
    min_sigma=MIN_SIGMA,
    run_closedloop_only=False,
    run_openloop_only=False,
)

FILTER_DATASETS = dict(annotated=True)

# None of these sessions is PZAH6.4b / PZAG3.4f, so the photodiode protocol is 5.
PHOTODIODE_PROTOCOL = 5

# The three targets. `file_special_sfx` lands in the pickle filename; `column_suffix` is
# applied later, at merge time, by merge_fit_dataframes.
#
# `column_suffix` is a per-method mapping rather than a single string that gets `_{method}`
# appended, because the settled convention is deliberately not uniform: bare `_treadmill`
# is the *per-frame plateau* family (it is read at ~240 hardcoded sites, so it keeps its
# name), while the trial-average family tags both methods explicitly. Spelling each target
# out here is what keeps a model merge from silently landing on plateau columns. See
# `revisions/migrate_treadmill_columns.py` for the full convention.
#
# `param_range` is per target rather than shared in COMMON_PARAMS: only the trial-average
# `treadmill` target is narrowed to the protocol grid. `sphere` samples running speed
# freely, so the wide default is correct for it, and `treadmill_frames` keeps the default
# so it stays comparable to the production per-frame `*_treadmill` columns it reproduces.
# Because every model in `initial_fit_conditions` reads the same four keys, one entry here
# covers gof/grs/gadd/gratio as well as g2d if they are ever re-run for that target.
FIT_TARGETS = {
    "sphere": dict(
        protocol_base="SpheresPermTubeReward",
        trial_average=False,
        max_rs2motor_diff=None,
        file_special_sfx="_legacy_refit",
        param_range=DEFAULT_PARAM_RANGE,
        # Closed-loop spheres never touch treadmill.sync_all_recordings, so `method` is
        # meaningless here and both entries are the untagged production columns.
        column_suffix={"model": "", "plateau": ""},
    ),
    "treadmill": dict(
        protocol_base="SpheresTubeMotor",
        trial_average=True,
        max_rs2motor_diff=0.3,
        file_special_sfx="_treadmill_trial_average_legacy",
        param_range=TREADMILL_PARAM_RANGE,
        column_suffix={
            "model": "_treadmill_trial_average_model",
            "plateau": "_treadmill_trial_average_plateau",
        },
    ),
    "treadmill_frames": dict(
        # Frame-by-frame SpheresTubeMotor fit, mirroring analysis_pipeline.py's own
        # per-frame treadmill config (protocol_base/max_rs2motor_diff/file_special_sfx
        # all match its `special_sfx_base = "_treadmill"` branch) so it's comparable to
        # the real, already-merged `*_treadmill` columns.
        protocol_base="SpheresTubeMotor",
        trial_average=False,
        max_rs2motor_diff=0.3,
        file_special_sfx="_treadmill_legacy",
        param_range=DEFAULT_PARAM_RANGE,
        # Plateau IS the bare `_treadmill` family - this target reproduces exactly what the
        # production pipeline now writes there - so it merges into it rather than beside it.
        column_suffix={"model": "_treadmill_model", "plateau": "_treadmill"},
    ),
}


# Expected processed-root prefix per site. The guard turns "cannot write to the wrong
# tree" into a property rather than a hope: on nemo the outputs land in the real session
# folders, and only the filename tags in FIT_TARGETS keep them from colliding with what is
# already there.
SITE_ROOTS = {
    "local": "/Volumes/BlackPasspo",
    "nemo": "/nemo/lab/znamenskiyp/home/shared/projects",
}


def assert_site_root(site):
    """Refuse to run unless this project's data root matches the declared site."""
    expected = SITE_ROOTS[site]
    root = Path(flz.get_data_root("processed", project=PROJECT))
    if not str(root).startswith(expected):
        raise RuntimeError(
            f"processed root for {PROJECT} is {root}, but --site {site} expects it to "
            f"start with {expected}. Refusing to run. Check the `project_paths` entry in "
            "~/.flexiznam/config.yml."
        )
    if not root.exists():
        raise RuntimeError(f"processed root {root} does not exist - is it mounted?")
    print(f"site={site}  processed root: {root}")
    print(f"          raw root: {flz.get_data_root('raw', project=PROJECT)}")


def target_config(target, method):
    """Per-(target, method) config: static FIT_TARGETS entry resolved for one onset method.

    The `sphere` target never touches `treadmill.sync_all_recordings` (it goes through
    `spheres.sync_all_recordings` instead), so `method` has no effect on its filename tag.

    *Filenames* keep the historical convention where `model` is untagged and other methods
    append `_{method}`, so the pickles already written to the session folders still match
    the merge glob: `..._treadmill_trial_average_legacy.pickle` (model) alongside
    `..._treadmill_trial_average_legacy_plateau.pickle` (plateau).

    *Columns* are resolved from the explicit `FIT_TARGETS[target]["column_suffix"]`
    mapping, never by string concatenation -- see the note on FIT_TARGETS for why the two
    axes are not tagged uniformly. This is the guard that stops a `model` merge from
    landing on the plateau columns that bare `_treadmill` now holds.
    """
    cfg = dict(FIT_TARGETS[target])
    if method not in cfg["column_suffix"]:
        raise ValueError(
            f"target {target!r} has no column_suffix for method {method!r}; "
            f"known: {sorted(cfg['column_suffix'])}"
        )
    cfg["column_suffix"] = cfg["column_suffix"][method]
    if cfg["protocol_base"] == "SpheresTubeMotor" and method != "model":
        cfg["file_special_sfx"] += f"_{method}"
    return cfg


def fit_filename(target, method, model, choose_trials, k_folds):
    """Output filename, matching what pipeline_utils.load_and_fit would produce."""
    suffix = (
        model + ("_crossval" if isinstance(choose_trials, str) else "") + f"_k{k_folds}"
    )
    special_sfx = target_config(target, method)["file_special_sfx"]
    return f"fit_rs_of_tuning_{suffix}{special_sfx}.pickle"


def _cottage_analysis_sha():
    """Short git SHA of the installed cottage_analysis, or None if it cannot be read."""
    import subprocess

    try:
        repo = Path(fit_gaussian_blob.__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return None


def param_range_record(target, method, cfg):
    """The fit parameters that a pickle's popts depend on but do not record.

    `fit_rs_of_tuning` stores `min_sigma` in the output (`rsof_minSigma_*`) but not
    `param_range`, `rs_thr` or `max_rs2motor_diff`. Since the g2d refit overwrites the
    `_treadmill_trial_average_plateau` pickles in place, without this record there is no
    way to tell afterwards which bounds produced a given file.
    """
    return {
        "target": target,
        "method": method,
        "param_range": cfg["param_range"],
        "min_sigma": COMMON_PARAMS["min_sigma"],
        "rs_thr": COMMON_PARAMS["rs_thr"],
        "niter": COMMON_PARAMS["niter"],
        "max_rs2motor_diff": cfg["max_rs2motor_diff"],
        "trial_average": cfg["trial_average"],
        "log_margin_sigma": float(_LOG_MARGIN),
        "cottage_analysis_sha": _cottage_analysis_sha(),
    }


def write_param_range_record(session_dir, target, method, cfg):
    """Write `param_range_current.json` next to the pickles this run produced.

    `.json` is invisible to `merge_fit_dataframes`' `fit_rs_of_tuning_*.pickle` glob, so
    this cannot be mistaken for a fit result to merge.
    """
    import json

    path = Path(session_dir) / "param_range_current.json"
    record = param_range_record(target, method, cfg)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"### wrote provenance record: {path}")
    return path


def report_rsof_range(trials_df, cfg, label=""):
    """Print the empirical RS/OF box of the data entering the fit, against `param_range`.

    This is what validates the protocol-grid assumption behind `TREADMILL_PARAM_RANGE`
    against real data: `param_range` bounds `x0`/`y0` in log space, so any sample outside
    the box is a stimulus condition the fit is no longer allowed to prefer.

    Reuses `figure_utils.trial_averages.reduce_trials_to_average`, which already
    replicates `fit_rs_of_tuning(trial_average=True)`'s running mask, rather than
    duplicating that mask here. Imported locally because `figure_utils` pulls in
    matplotlib, which this script otherwise does not need.

    Returns:
        dict or None: `n`, `rs_min`/`rs_max` (m/s), `of_min`/`of_max` (deg/s) and
            `frac_outside`; None if the target is not trial-averaged (the reduction does
            not apply) or no trial survived the mask.
    """
    from v1_depth_map.figure_utils import trial_averages

    if not cfg["trial_average"]:
        print(
            f"--- {label}: --check-range only covers trial-averaged targets "
            "(reduce_trials_to_average replicates that path); skipping."
        )
        return None

    reduced = trial_averages.reduce_trials_to_average(
        trials_df,
        rs_thr=COMMON_PARAMS["rs_thr"],
        max_rs2motor_diff=cfg["max_rs2motor_diff"],
        closed_loop_only=True,
    )
    if reduced is None:
        print(f"--- {label}: no trial survived the running mask")
        return None

    pr = cfg["param_range"]
    rs = np.asarray(reduced["rs"], dtype=float)  # m/s
    of = np.degrees(
        np.asarray(reduced["of"], dtype=float)
    )  # rad/s -> deg/s, as the fit
    outside = (
        (rs < pr["rs_min"])
        | (rs > pr["rs_max"])
        | (of < pr["of_min"])
        | (of > pr["of_max"])
    )
    stats = dict(
        n=int(rs.size),
        rs_min=float(rs.min()),
        rs_max=float(rs.max()),
        of_min=float(of.min()),
        of_max=float(of.max()),
        frac_outside=float(np.mean(outside)),
    )
    print(f"--- {label}: {stats['n']} trial averages")
    print(
        f"      RS  data [{stats['rs_min']:.4f}, {stats['rs_max']:.4f}] m/s   "
        f"bounds [{pr['rs_min']:.4f}, {pr['rs_max']:.4f}]"
    )
    print(
        f"      OF  data [{stats['of_min']:.3f}, {stats['of_max']:.1f}] deg/s   "
        f"bounds [{pr['of_min']:.3f}, {pr['of_max']:.1f}]"
    )
    if outside.any():
        print(
            f"      WARNING: {outside.sum()}/{rs.size} "
            f"({100 * stats['frac_outside']:.1f}%) samples fall OUTSIDE param_range - "
            "the bounds are narrower than the stimulus. Widen them before fitting."
        )
    else:
        print("      OK: every sample is inside param_range")
    return stats


def _session_dir_or_none(session_name):
    """The session's processed folder, resolved via flexilims without loading the session.

    Same cheap lookup `submit_one` uses for its `skip_existing` check. Returns None rather
    than raising, so a dry run still lists its configs if flexilims is unreachable -- in
    the spirit of `_count_existing_target_columns`: a reporting helper must never be the
    thing that fails the operation.
    """
    try:
        flm = flz.get_flexilims_session(project_id=PROJECT)
        return pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flm,
            project=PROJECT,
            conflicts="skip",
        ).path_full.parent
    except Exception as exc:  # noqa: BLE001 - reporting only
        print(f"  (could not resolve session folder for {session_name}: {exc})")
        return None


def fit_session_target(
    session_name,
    target,
    method,
    configs,
    dry_run=False,
    skip_existing=True,
    check_range_only=False,
):
    """Fit every config of one (session, target), loading the session data ONCE.

    This deliberately inlines what `pipeline_utils.load_and_fit` does rather than calling
    it once per config, because `load_and_fit` re-runs `load_session` every time. On a
    16 GB machine the rebuilt `trials_df` (~50k frames x ~800 ROIs of dF/F) plus the
    200 MB `neurons_df` is the dominant memory cost and the load is pure overhead when
    the 11 configs all consume the same `trials_df`. Keep this in step with
    `load_and_fit` (pipeline_utils.py:246) if that changes.
    """
    cfg = target_config(target, method)

    if dry_run:
        print(f"[dry-run] {session_name} | {target} | param_range={cfg['param_range']}")
        # Resolve the output folder without loading the session, the way submit_one does,
        # so a dry run can report what a *resume* would actually do rather than just
        # listing every config. This is what makes `--dry-run` usable as a resume-status
        # view: an accidental `--redo` shows up as "would fit" on already-finished configs.
        session_dir = _session_dir_or_none(session_name)
        n_skip = 0
        for model, choose_trials, k_folds in configs:
            fname = fit_filename(target, method, model, choose_trials, k_folds)
            if session_dir is None:
                verdict = "?         "  # could not resolve; still list the config
            elif skip_existing and (session_dir / fname).exists():
                verdict = "skip (exists)"
                n_skip += 1
            else:
                verdict = "would fit"
            print(
                f"[dry-run] {session_name} | {target:16s} | {model} "
                f"choose_trials={choose_trials} k={k_folds}"
                f"\n          {verdict:>13s} -> {fname}"
            )
        if session_dir is not None:
            print(
                f"[dry-run] {session_name} | {target}: {len(configs) - n_skip} to fit, "
                f"{n_skip} already done"
                + ("" if skip_existing else "  (--redo: nothing will be skipped)")
            )
        return []

    what = "range check" if check_range_only else f"{len(configs)} configs"
    print(f"\n### {session_name} | {target} | loading session once for {what}")
    print(f"### param_range: {cfg['param_range']}")
    t_load = time.time()
    neurons_ds, _, _, trials_df_all = pipeline_utils.load_session(
        project=PROJECT,
        session_name=session_name,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        regenerate_frames=False,
        filter_datasets=FILTER_DATASETS,
        protocol_base=cfg["protocol_base"],
        recording_type="two_photon",
        tread_kwargs={"method": method},
    )
    # Drop multidepth recordings, as load_and_fit does (pipeline_utils.py:352-354).
    is_multidepth = trials_df_all.recording_name.str.contains("multidepth")
    trials_df_all = trials_df_all[~is_multidepth]
    print(
        f"### loaded in {(time.time() - t_load) / 60:.1f} min "
        f"({len(trials_df_all)} trials)"
    )

    # Always report the empirical box, so every real run records whether the bounds it used
    # actually contain the stimulus it fit.
    report_rsof_range(trials_df_all, cfg, label=f"{session_name} | {target}")
    if check_range_only:
        return []

    write_param_range_record(neurons_ds.path_full.parent, target, method, cfg)

    durations = []
    for model, choose_trials, k_folds in configs:
        fname = fit_filename(target, method, model, choose_trials, k_folds)
        out_path = neurons_ds.path_full.with_name(fname)
        if skip_existing and out_path.exists():
            print(f"--- skip (exists): {fname}")
            continue
        print(
            f"\n=== {session_name} | {target} | {model} "
            f"choose_trials={choose_trials} k={k_folds}"
        )
        t0 = time.time()
        fit_df = fit_gaussian_blob.fit_rs_of_tuning(
            trials_df=trials_df_all,
            model=model,
            choose_trials=choose_trials,
            trial_sfx="",
            k_folds=k_folds,
            max_rs2motor_diff=cfg["max_rs2motor_diff"],
            trial_average=cfg["trial_average"],
            param_range=cfg["param_range"],
            **COMMON_PARAMS,
        )
        # Write to a temp file and rename. `rename` is atomic within a filesystem, so an
        # interrupted run can never leave a truncated pickle at `out_path` — which matters
        # because `skip_existing` would otherwise treat a half-written file as complete.
        tmp = out_path.with_suffix(".pickle.partial")
        fit_df.to_pickle(tmp)
        tmp.replace(out_path)
        dt = time.time() - t0
        durations.append(dt)
        print(f"=== done in {dt / 60:.1f} min -> {fname}")
    return durations


def submit_one(
    session_name,
    target,
    method,
    model,
    choose_trials,
    k_folds,
    slurm_root,
    dry_run=False,
    skip_existing=True,
):
    """Submit ONE (session, target, config) fit as a slurm job.

    Unlike `fit_session_target`, this goes through `pipeline_utils.load_and_fit`, which
    reloads the session itself. That redundant load is deliberate here: on the cluster
    every config runs as its own job, so maximum parallelism beats sharing a `trials_df`.
    """
    cfg = target_config(target, method)
    fname = fit_filename(target, method, model, choose_trials, k_folds)
    label = (
        f"{session_name} | {target:16s} | {model} choose_trials={choose_trials} "
        f"k={k_folds}"
    )

    if skip_existing:
        flm = flz.get_flexilims_session(project_id=PROJECT)
        out_path = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flm,
            project=PROJECT,
            conflicts="skip",
        ).path_full.with_name(fname)
        if out_path.exists():
            print(f"--- skip (exists): {fname}")
            return None

    # scripts_name must be unique per job: slurm_it writes {slurm_folder}/{scripts_name}.py
    # and .sh, so two jobs sharing a name overwrite each other's scripts.
    trials_tag = f"_{choose_trials}" if isinstance(choose_trials, str) else ""
    scripts_name = f"refit_{session_name}_{target}_{model}{trials_tag}_k{k_folds}"
    slurm_folder = Path(slurm_root) / session_name

    if dry_run:
        print(
            f"[dry-run] submit {label}\n          -> {fname}\n"
            f"          scripts_name={scripts_name}\n"
            f"          param_range={cfg['param_range']}"
        )
        return None

    print(f"    param_range: {cfg['param_range']}")
    slurm_folder.mkdir(parents=True, exist_ok=True)  # slurm_it asserts it exists
    job_id = pipeline_utils.load_and_fit(
        project=PROJECT,
        session_name=session_name,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        model=model,
        choose_trials=choose_trials,
        k_folds=k_folds,
        protocol_base=cfg["protocol_base"],
        trial_average=cfg["trial_average"],
        max_rs2motor_diff=cfg["max_rs2motor_diff"],
        param_range=cfg["param_range"],
        file_special_sfx=cfg["file_special_sfx"],
        trial_sfx="",
        filter_datasets=FILTER_DATASETS,
        tread_kwargs={"method": method},
        use_slurm=True,
        slurm_folder=str(slurm_folder),
        scripts_name=scripts_name,
        # Measured on ncpu (2026-08-05 run), sphere per-frame, per config:
        #   4-param models (gof/grs/gratio): k1 12-25 min,   k5 55 min - 2 h
        #   7-param models (g2d/gadd):       k1 2.2 - 5 h,   k5 10 - 24 h
        # The 7-param fits are an order of magnitude slower than the earlier local
        # estimate, so the limits are set from these numbers with headroom rather than
        # from that estimate. ncpu allows 7 days; peak RSS is 1.7 GB against the 32 G
        # requested, so memory is not the constraint. This budget is applied to every
        # target unconditionally, including `treadmill_frames` - its SpheresTubeMotor
        # recordings are shorter than the sphere sessions, so these sphere numbers are a
        # conservative upper bound for it, not a tight estimate. Time one config first
        # (see the module docstring) before trusting that headroom for the full set.
        slurm_options={
            "mem": "32G",
            "time": "48:00:00" if k_folds > 1 else "12:00:00",
            "partition": "ncpu",
            "cpus-per-task": 8,
        },
        **COMMON_PARAMS,
    )
    print(f"submitted job {job_id}  {label}")
    return job_id


def _count_existing_target_columns(session_name, column_suffix):
    """How many columns in this session's neurons_df already end in `column_suffix`.

    Returns 0 (rather than raising) if the file cannot be read, so a reporting helper can
    never be the thing that fails a merge.
    """
    if not column_suffix:
        return 0
    try:
        flm = flz.get_flexilims_session(project_id=PROJECT)
        neurons_ds = flz.get_datasets(
            origin_name=session_name,
            dataset_type="neurons_df",
            flexilims_session=flm,
            allow_multiple=False,
        )
        cols = pd.read_pickle(neurons_ds.path_full).columns
    except Exception as exc:  # noqa: BLE001 - reporting only
        print(f"  (could not pre-read neurons_df to check for collisions: {exc})")
        return 0
    return sum(1 for c in cols if c.endswith(column_suffix))


def merge_one(session_name, target, method, conflicts=None, dry_run=False):
    """Merge one target's fit pickles into the session's neurons_df.pickle."""
    cfg = target_config(target, method)
    # All three targets carry a filename tag, so the glob f"{prefix}*{suffix}{filetype}" is
    # specific to one target and cannot pick up another's files. (With an untagged sphere
    # target it could, and "treadmill" would have to be in exclude_keywords to prevent it.)
    exclude_keywords = ["recording", "openclosed", "openloop"]
    if target == "sphere":
        # The sphere columns already exist in neurons_df, and conflicts="skip" only adds
        # columns that are absent — it would silently merge nothing.
        default_conflicts = "overwrite"
    else:
        # A method-tagged treadmill target is normally new, so "skip" cannot clobber
        # anything. The exception is `treadmill_frames` + plateau, whose target is the
        # pre-existing bare `_treadmill` family; that case is caught below.
        default_conflicts = "skip"

    if conflicts is None:
        conflicts = default_conflicts

    # merge_fit_dataframes with conflicts="skip" adds only columns that are ABSENT, so a
    # merge onto an already-populated family is a silent no-op. Say so before doing it,
    # rather than reporting success and changing nothing.
    n_existing = _count_existing_target_columns(session_name, cfg["column_suffix"])
    if n_existing and conflicts == "skip":
        print(
            f"WARNING: {n_existing} columns already end in {cfg['column_suffix']!r} and "
            f"conflicts='skip' — this merge will write NOTHING. Pass "
            f"--conflicts overwrite to replace them."
        )

    label = (
        f"{session_name} | {target:16s} | suffix={cfg['file_special_sfx']!r} "
        f"-> columns{cfg['column_suffix']!r} (conflicts={conflicts}, "
        f"{n_existing} already present)"
    )
    if dry_run:
        print(f"[dry-run] merge {label}")
        return None

    print(f"\n=== merge {label}")
    return pipeline_utils.merge_fit_dataframes(
        project=PROJECT,
        session_name=session_name,
        conflicts=conflicts,
        prefix="fit_rs_of_tuning_",
        suffix=cfg["file_special_sfx"],
        exclude_keywords=exclude_keywords,
        target_column_suffix=cfg["column_suffix"],
        filetype=".pickle",
        target_filename="neurons_df.pickle",
        use_slurm=False,
    )


def parse_config(spec):
    """Parse a "model:choose_trials:k_folds" string, e.g. "gaussian_2d:None:1"."""
    model, trials, k = spec.split(":")
    return model, (None if trials in ("None", "none", "") else trials), int(k)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sessions", nargs="+", default=SESSIONS)
    parser.add_argument(
        "--only",
        choices=sorted(FIT_TARGETS),
        default=None,
        help="Run only one fit target.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help='Configs as "model:choose_trials:k_folds", e.g. gaussian_2d:None:1.',
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing fit pickles into neurons_df.pickle instead of fitting.",
    )
    parser.add_argument(
        "--conflicts",
        default=None,
        choices=["skip", "overwrite"],
        help="Override the per-target default merge conflict policy.",
    )
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Re-fit configs whose output pickle already exists (default: skip them).",
    )
    parser.add_argument(
        "--method",
        choices=["model", "plateau"],
        default="model",
        help="Onset-detection method passed to treadmill.sync_all_recordings for any "
        "SpheresTubeMotor target ('treadmill', 'treadmill_frames'; ignored by the "
        "'sphere' target). 'plateau' tags output filenames/columns with a '_plateau' "
        "suffix so they don't clobber the existing 'model' results. Defaults to "
        "'model'.",
    )
    parser.add_argument(
        "--site",
        choices=sorted(SITE_ROOTS),
        default="local",
        help="Which data root to expect; refuses to run if it does not match.",
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        help="Submit one slurm job per (session, target, config) instead of fitting inline.",
    )
    parser.add_argument(
        "--slurm-folder",
        default=str(Path.home() / "slurm_logs" / "legacy_refit"),
        help="Root for slurm scripts and logs; a per-session subfolder is created.",
    )
    parser.add_argument(
        "--check-range",
        action="store_true",
        help="Read-only: load each session and report the empirical trial-averaged RS/OF "
        "box against the target's param_range, then stop without fitting. Run this "
        "before a refit to confirm the bounds actually contain the stimulus.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assert_site_root(args.site)

    targets = [args.only] if args.only else list(FIT_TARGETS)
    configs = [parse_config(c) for c in args.configs] if args.configs else MODEL_CONFIGS

    if args.check_range:
        for session_name in args.sessions:
            for target in targets:
                fit_session_target(
                    session_name,
                    target,
                    args.method,
                    configs,
                    dry_run=args.dry_run,
                    check_range_only=True,
                )
        return

    if args.merge:
        for session_name in args.sessions:
            for target in targets:
                merge_one(
                    session_name,
                    target,
                    args.method,
                    args.conflicts,
                    dry_run=args.dry_run,
                )
        return

    total = len(args.sessions) * len(targets) * len(configs)
    print(
        f"\n{total} fits: {len(args.sessions)} sessions x {len(targets)} targets x "
        f"{len(configs)} configs "
        f"({'re-fitting existing' if args.redo else 'skipping existing'})"
    )

    if args.use_slurm:
        job_ids = []
        for session_name in args.sessions:
            for target in targets:
                for model, choose_trials, k_folds in configs:
                    jid = submit_one(
                        session_name,
                        target,
                        args.method,
                        model,
                        choose_trials,
                        k_folds,
                        args.slurm_folder,
                        dry_run=args.dry_run,
                        skip_existing=not args.redo,
                    )
                    if jid:
                        job_ids.append(jid)
        print(f"\n{len(job_ids)} jobs submitted. Track with: squeue -u $USER")
        if job_ids:
            print("job ids: " + " ".join(str(j) for j in job_ids))
        return

    durations = []
    for session_name in args.sessions:
        for target in targets:
            durations += fit_session_target(
                session_name,
                target,
                args.method,
                configs,
                dry_run=args.dry_run,
                skip_existing=not args.redo,
            )
            if durations:
                print(
                    f"    [{len(durations)}/{total} fitted] cumulative "
                    f"{sum(durations) / 60:.1f} min"
                )

    if durations:
        print(
            f"\nAll done: {len(durations)} fits in {sum(durations) / 60:.1f} min "
            f"(mean {sum(durations) / len(durations) / 60:.1f} min/fit)"
        )


if __name__ == "__main__":
    main()
