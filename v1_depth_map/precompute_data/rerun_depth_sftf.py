"""Rerun the grating (SFTF) fit of one session, overwriting its neurons_df_sftf.pickle.

Runs `cottage_analysis.pipelines.depth_SFTF.main` (sync of the SFTF recordings with the
`ast_neuropil=False` traces, single-trial `fit_sftf_tuning`, by default cross-validated with
one fold per recording), after backing up the existing `neurons_df_sftf.pickle`:

- the first run copies it to BACKUP_DIR/SESSION/neurons_df_sftf.pickle and writes its md5
  next to it (`neurons_df_sftf.pickle.md5`);
- every run checks that the backup still has that md5 and that the file to overwrite still
  matches it, and stops otherwise (so a second run cannot overwrite a new fit; restore or
  delete the backup to run again).

Provenance (k_folds, traces filter, onset volumes skipped in each trial, backup) is saved as
`neurons_df_sftf.json` next to the pickle. Restore with
`cp -p BACKUP_DIR/SESSION/neurons_df_sftf.pickle <session folder>/`. To overwrite a file that this
script fitted itself, use a new BACKUP_DIR (the md5 check is against the backup in BACKUP_DIR).

With `--use-col spks`, the deconvolved spikes are fitted instead and saved as a new
`neurons_df_sftf_spks.pickle` (+ .json); nothing is backed up or overwritten, and the run
stops if that file exists.

Usage:
    python rerun_depth_sftf.py SESSION --backup-dir DIR [--project P] [--k-folds 5] [--n-jobs 1]
        [--use-col {dffs,spks}]
"""

import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import flexiznam as flz

from cottage_analysis.analysis import gratings
from cottage_analysis.pipelines import depth_SFTF, pipeline_utils

PROJECT = "hey2_3d-vision_foodres_20220101"
FILENAME = "neurons_df_sftf.pickle"


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def backup_and_check(target, backup_dir):
    """Back up `target` on the first call, then check that backup and target are unchanged.

    Returns:
        str: md5 of the backed-up file.
    """
    backup = backup_dir / FILENAME
    md5_file = backup_dir / f"{FILENAME}.md5"
    if not backup.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        md5_file.write_text(f"{md5(target)}  {FILENAME}\n")
        print(f"backed up {target} -> {backup}")
    recorded = md5_file.read_text().split()[0]
    assert md5(backup) == recorded, f"{backup} does not match {md5_file}"
    assert md5(target) == recorded, (
        f"{target} changed since it was backed up to {backup}: not overwriting. "
        "Restore or delete the backup to run again."
    )
    return recorded


def main(
    session, backup_dir, project, k_folds, n_jobs, photodiode_protocol, use_col="dffs"
):
    flm = flz.get_flexilims_session(project)
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=flm, project=project, conflicts="skip"
    )
    if use_col == "dffs":
        target = neurons_ds.path_full.parent / FILENAME
        backup_md5 = backup_and_check(target, Path(backup_dir) / session)
        print(f"{session}: backup verified ({backup_md5}), rerunning depth_SFTF")
        provenance = dict(
            previous_md5=backup_md5, backup=str(Path(backup_dir) / session / FILENAME)
        )
    else:
        target = neurons_ds.path_full.parent / f"neurons_df_sftf_{use_col}.pickle"
        assert not target.exists(), f"{target} exists: not overwriting"
        print(f"{session}: fitting depth_SFTF on {use_col}")
        provenance = {}

    depth_SFTF.main(
        project,
        session,
        conflicts="skip",  # only resolves the neurons_df dataset; nothing is skipped
        photodiode_protocol=photodiode_protocol,
        k_folds=k_folds,
        n_jobs=n_jobs,
        use_col=use_col,
    )
    target.with_suffix(".json").write_text(
        json.dumps(
            dict(
                k_folds=k_folds,
                fold_col="irecording",
                filter_traces={"anatomical_only": 3, "ast_neuropil": False},
                photodiode_protocol=photodiode_protocol,
                use_col=use_col,
                skip_first_n_volumes=inspect.signature(gratings.generate_trials_df)
                .parameters["skip_first_n_volumes"]
                .default,
                **provenance,
            ),
            indent=1,
        )
    )
    print(f"Saved {target} (md5 {md5(target)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument(
        "--backup-dir", required=True, help="backups go to BACKUP_DIR/SESSION/"
    )
    p.add_argument("--project", default=PROJECT)
    p.add_argument("--k-folds", type=int, default=5, help="1 for no cross-validation")
    p.add_argument("--n-jobs", type=int, default=1, help="ROIs fitted in parallel")
    p.add_argument("--photodiode", type=int, default=5)
    p.add_argument(
        "--use-col",
        choices=["dffs", "spks"],
        default="dffs",
        help="spks: fit deconvolved spikes into neurons_df_sftf_spks.pickle",
    )
    a = p.parse_args()
    main(
        a.session,
        a.backup_dir,
        a.project,
        a.k_folds,
        a.n_jobs,
        a.photodiode,
        a.use_col,
    )
