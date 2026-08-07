"""Re-run the RS/OF tuning fits for the revision treadmill sessions.

The four `colasa_3d-vision_revisions` sessions that have a `SpheresTubeMotor` recording
are fit twice:

- the **sphere** (closed-loop `SpheresPermTubeReward`) part **frame by frame**, as the
  standard pipeline does, writing `fit_rs_of_tuning_<model>[_crossval]_k<n>.pickle`;
- the **treadmill** (`SpheresTubeMotor`) part on **trial averages**, writing
  `fit_rs_of_tuning_<model>[_crossval]_k<n>_treadmill_trial_average_legacy.pickle`.

Both use the *legacy* ``(log_sigma_x2, log_sigma_y2, theta)`` 2D-Gaussian parameterisation
of the `reviews` branch of cottage_analysis. That is the point of the `_legacy` filename
tag: nemo carries identically-named `..._treadmill_trial_average.pickle` files whose popts
are in the newer Cholesky parameterisation, and the two are NOT interchangeable. The tag
makes them impossible to confuse if anything is ever synced between the two locations.

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

    python fit_revision_treadmill.py              # all sessions, both halves, 11 configs
    python fit_revision_treadmill.py --merge      # merge results into neurons_df.pickle

`--method plateau` re-fits the treadmill half with `treadmill.sync_all_recordings`'s
trapezoidal-ramp onset detector instead of the default `"model"` heuristic (the `sphere`
half is unaffected, since it never goes through `treadmill.sync_all_recordings`). Output
filenames/columns get a `_plateau` tag so they land alongside the `"model"` results instead
of overwriting them::

    python fit_revision_treadmill.py --only treadmill --method plateau
    python fit_revision_treadmill.py --only treadmill --method plateau --merge
"""

import argparse
import time
from pathlib import Path

import flexiznam as flz

from cottage_analysis.analysis import fit_gaussian_blob
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

# Shared fit parameters, matching analysis_pipeline's `common_params`.
COMMON_PARAMS = dict(
    rs_thr=0.01,
    param_range={"rs_min": 0.005, "rs_max": 5, "of_min": 0.03, "of_max": 3000},
    niter=10,
    min_sigma=0.25,
    run_closedloop_only=False,
    run_openloop_only=False,
)

FILTER_DATASETS = dict(annotated=True)

# None of these sessions is PZAH6.4b / PZAG3.4f, so the photodiode protocol is 5.
PHOTODIODE_PROTOCOL = 5

# The two halves. `file_special_sfx` lands in the pickle filename; `column_suffix` is
# applied later, at merge time, by merge_fit_dataframes.
HALVES = {
    "sphere": dict(
        protocol_base="SpheresPermTubeReward",
        trial_average=False,
        max_rs2motor_diff=None,
        file_special_sfx="_legacy_refit",
        column_suffix="",
    ),
    "treadmill": dict(
        protocol_base="SpheresTubeMotor",
        trial_average=True,
        max_rs2motor_diff=0.3,
        file_special_sfx="_treadmill_trial_average_legacy",
        column_suffix="_treadmill_trial_average",
    ),
}


# Expected processed-root prefix per site. The guard turns "cannot write to the wrong
# tree" into a property rather than a hope: on nemo the outputs land in the real session
# folders, and only the filename tags in HALVES keep them from colliding with what is
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


def half_config(half, method):
    """Per-(half, method) config: static HALVES entry, with filename/column tags for
    non-default methods.

    The `sphere` half never touches `treadmill.sync_all_recordings` (it goes through
    `spheres.sync_all_recordings` instead), so `method` has no effect on it and its
    tags are left untouched regardless of what is passed in. Only the `treadmill`
    half's tags grow a `_{method}` suffix when `method != "model"`, so that e.g.
    `method="plateau"` writes to `..._treadmill_trial_average_legacy_plateau.pickle`
    and merges into `..._treadmill_trial_average_plateau` columns -- distinct from the
    existing `model`-method pickles/columns, so neither run clobbers the other.
    """
    cfg = dict(HALVES[half])
    if half == "treadmill" and method != "model":
        cfg["file_special_sfx"] += f"_{method}"
        cfg["column_suffix"] += f"_{method}"
    return cfg


def fit_filename(half, method, model, choose_trials, k_folds):
    """Output filename, matching what pipeline_utils.load_and_fit would produce."""
    suffix = (
        model + ("_crossval" if isinstance(choose_trials, str) else "") + f"_k{k_folds}"
    )
    return f"fit_rs_of_tuning_{suffix}{half_config(half, method)['file_special_sfx']}.pickle"


def fit_session_half(
    session_name, half, method, configs, dry_run=False, skip_existing=True
):
    """Fit every config of one (session, half), loading the session data ONCE.

    This deliberately inlines what `pipeline_utils.load_and_fit` does rather than calling
    it once per config, because `load_and_fit` re-runs `load_session` every time. On a
    16 GB machine the rebuilt `trials_df` (~50k frames x ~800 ROIs of dF/F) plus the
    200 MB `neurons_df` is the dominant memory cost and the load is pure overhead when
    the 11 configs all consume the same `trials_df`. Keep this in step with
    `load_and_fit` (pipeline_utils.py:246) if that changes.
    """
    cfg = half_config(half, method)

    if dry_run:
        for model, choose_trials, k_folds in configs:
            print(
                f"[dry-run] {session_name} | {half:9s} | {model} "
                f"choose_trials={choose_trials} k={k_folds}"
                f"\n          -> {fit_filename(half, method, model, choose_trials, k_folds)}"
            )
        return []

    print(
        f"\n### {session_name} | {half} | loading session once for {len(configs)} configs"
    )
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

    durations = []
    for model, choose_trials, k_folds in configs:
        fname = fit_filename(half, method, model, choose_trials, k_folds)
        target = neurons_ds.path_full.with_name(fname)
        if skip_existing and target.exists():
            print(f"--- skip (exists): {fname}")
            continue
        print(
            f"\n=== {session_name} | {half} | {model} "
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
            **COMMON_PARAMS,
        )
        # Write to a temp file and rename. `rename` is atomic within a filesystem, so an
        # interrupted run can never leave a truncated pickle at `target` — which matters
        # because `skip_existing` would otherwise treat a half-written file as complete.
        tmp = target.with_suffix(".pickle.partial")
        fit_df.to_pickle(tmp)
        tmp.replace(target)
        dt = time.time() - t0
        durations.append(dt)
        print(f"=== done in {dt / 60:.1f} min -> {fname}")
    return durations


def submit_one(
    session_name,
    half,
    method,
    model,
    choose_trials,
    k_folds,
    slurm_root,
    dry_run=False,
    skip_existing=True,
):
    """Submit ONE (session, half, config) fit as a slurm job.

    Unlike `fit_session_half`, this goes through `pipeline_utils.load_and_fit`, which
    reloads the session itself. That redundant load is deliberate here: on the cluster
    every config runs as its own job, so maximum parallelism beats sharing a `trials_df`.
    """
    cfg = half_config(half, method)
    fname = fit_filename(half, method, model, choose_trials, k_folds)
    label = (
        f"{session_name} | {half:9s} | {model} choose_trials={choose_trials} "
        f"k={k_folds}"
    )

    if skip_existing:
        flm = flz.get_flexilims_session(project_id=PROJECT)
        target = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flm,
            project=PROJECT,
            conflicts="skip",
        ).path_full.with_name(fname)
        if target.exists():
            print(f"--- skip (exists): {fname}")
            return None

    # scripts_name must be unique per job: slurm_it writes {slurm_folder}/{scripts_name}.py
    # and .sh, so two jobs sharing a name overwrite each other's scripts.
    trials_tag = f"_{choose_trials}" if isinstance(choose_trials, str) else ""
    scripts_name = f"refit_{session_name}_{half}_{model}{trials_tag}_k{k_folds}"
    slurm_folder = Path(slurm_root) / session_name

    if dry_run:
        print(
            f"[dry-run] submit {label}\n          -> {fname}\n"
            f"          scripts_name={scripts_name}"
        )
        return None

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
        # requested, so memory is not the constraint.
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


def merge_one(session_name, half, method, conflicts=None, dry_run=False):
    """Merge one half's fit pickles into the session's neurons_df.pickle."""
    cfg = half_config(half, method)
    # Both halves carry a filename tag, so the glob f"{prefix}*{suffix}{filetype}" is
    # specific to one half and cannot pick up the other's files. (With an untagged sphere
    # half it could, and "treadmill" would have to be in exclude_keywords to prevent it.)
    exclude_keywords = ["recording", "openclosed", "openloop"]
    if half == "sphere":
        # The sphere columns already exist in neurons_df, and conflicts="skip" only adds
        # columns that are absent — it would silently merge nothing.
        default_conflicts = "overwrite"
    else:
        # Every resulting column is new, so "skip" is correct and cannot clobber anything.
        default_conflicts = "skip"

    if conflicts is None:
        conflicts = default_conflicts

    label = (
        f"{session_name} | {half:9s} | suffix={cfg['file_special_sfx']!r} "
        f"-> columns{cfg['column_suffix']!r} (conflicts={conflicts})"
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
        "--only", choices=sorted(HALVES), default=None, help="Run only one half."
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
        help="Override the per-half default merge conflict policy.",
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
        help="Onset-detection method passed to treadmill.sync_all_recordings for the "
        "'treadmill' half (ignored by the 'sphere' half). 'plateau' tags output "
        "filenames/columns with a '_plateau' suffix so they don't clobber the "
        "existing 'model' results. Defaults to 'model'.",
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
        help="Submit one slurm job per (session, half, config) instead of fitting inline.",
    )
    parser.add_argument(
        "--slurm-folder",
        default=str(Path.home() / "slurm_logs" / "legacy_refit"),
        help="Root for slurm scripts and logs; a per-session subfolder is created.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    assert_site_root(args.site)

    halves = [args.only] if args.only else list(HALVES)
    configs = [parse_config(c) for c in args.configs] if args.configs else MODEL_CONFIGS

    if args.merge:
        for session_name in args.sessions:
            for half in halves:
                merge_one(
                    session_name,
                    half,
                    args.method,
                    args.conflicts,
                    dry_run=args.dry_run,
                )
        return

    total = len(args.sessions) * len(halves) * len(configs)
    print(
        f"\n{total} fits: {len(args.sessions)} sessions x {len(halves)} halves x "
        f"{len(configs)} configs "
        f"({'re-fitting existing' if args.redo else 'skipping existing'})"
    )

    if args.use_slurm:
        job_ids = []
        for session_name in args.sessions:
            for half in halves:
                for model, choose_trials, k_folds in configs:
                    jid = submit_one(
                        session_name,
                        half,
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
        for half in halves:
            durations += fit_session_half(
                session_name,
                half,
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
