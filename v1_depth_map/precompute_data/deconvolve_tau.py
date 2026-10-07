"""Deconvolve the suite2p traces of one session again with a new tau.

Runs the deconvolution step of the 2p-preprocess pipeline
(`spike_deconvolution_suite2p` on `Fstandard.npy`, `ast_neuropil=False`) on the
concatenated suite2p dataset (annotated for the revision project), then splits `spks.npy` per recording with
the indexing of `split_recordings`. So the spikes are the same as a rerun of the
pipeline with this tau. Only `spks.npy` is written; F, Fneu and dF/F are not.

The split is checked by applying the same indexing to the concatenated `dff.npy`,
which must reproduce the `dff.npy` of every split dataset.

The concatenated dataset is the `suite2p_rois` dataset of the session with
`Fstandard.npy` whose split indexing reproduces the split traces (parent datasets may
carry stale attribute labels).

Modes:
    --check-only: recompute with --tau and compare with the stored spks.npy, no writes.
    default: overwrite `spks.npy` in the processed folders (concatenated and split),
        after copying the current files to `--backup-dir` (once). With
        `--set-tau-attribute`, also set the `tau` attribute of the datasets on flexilims.
    --out-dir DIR: write the spikes to DIR/<recording>/plane*/spks.npy (and the
        concatenated ones to DIR/concatenated/plane*/spks.npy), processed data untouched.
        DIR/paths.json maps each folder to the split dataset path.

Split datasets of a session share their name, so files are kept per recording
folder (the parent folder of the split dataset).

Needs the `2p-preprocess` environment (suite2p).

Usage:
    python deconvolve_tau.py SESSION --tau 1.1 --backup-dir DIR
    python deconvolve_tau.py SESSION --tau 0.7 --check-only [--project hey2_3d-vision_foodres_20220101]
"""

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import flexiznam as flz
from suite2p.extraction import dcnv
from twop_preprocess.calcium.calcium_s2p import spike_deconvolution_suite2p
from twop_preprocess.calcium.calcium_utils import get_recording_frames

PROJECT = "colasa_3d-vision_revisions"
FILTER_TRACES = {"anatomical_only": 3, "annotated": True, "ast_neuropil": False}
# split traces used by the analyses of each project (and datasets to exclude)
TRACES = {
    "colasa_3d-vision_revisions": (FILTER_TRACES, None),
    "hey2_3d-vision_foodres_20220101": (
        {"anatomical_only": 3, "ast_neuropil": False},
        {"annotated": True},
    ),
}


def concatenated_dataset(fs, session, project=PROJECT):
    """The concatenated suite2p_rois dataset whose split indexing gives the analysed traces.

    Returns:
        tuple: the dataset and its split targets (see `split_targets`).
    """
    filt, excl = TRACES[project]
    ds = flz.get_datasets(
        flexilims_session=fs,
        origin_name=session,
        dataset_type="suite2p_rois",
        return_dataseries=True,
    )
    found = []
    for _, d in ds.iterrows():
        cand = flz.Dataset.from_dataseries(d, fs)
        if float(cand.extra_attributes.get("anatomical_only", -1)) != 3:
            continue
        if bool(cand.extra_attributes.get("annotated", False)) != bool(
            filt.get("annotated", False)
        ):
            continue
        if not (cand.path_full / "plane0" / "Fstandard.npy").exists():
            continue
        try:
            targets = split_targets(fs, cand, project)
            check_split(cand, targets)
        except (AssertionError, FileNotFoundError, ValueError) as err:
            print(
                f"  {cand.dataset_name}: not the parent of the analysed traces ({err})"
            )
            continue
        found.append((cand, targets))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{len(found)} suite2p_rois datasets of {session} reproduce the split traces"
        )
    return found[0]


def check_split(suite2p_ds, targets):
    """The split indexing applied to the concatenated dff.npy must give each split dff.npy."""
    nplanes = int(float(suite2p_ds.extra_attributes["nplanes"]))
    for iplane in range(nplanes):
        dff = np.load(suite2p_ds.path_full / f"plane{iplane}" / "dff.npy")
        if dff.shape[1] == 0:
            continue
        for recording_id, first, nframes, split in targets:
            stored = np.load(split.path_full / f"plane{iplane}" / "dff.npy")
            start = first[iplane]
            assert np.array_equal(
                dff[:, start : start + nframes], stored, equal_nan=True
            ), f"split indexing does not reproduce dff of {rec_folder(split)} plane {iplane}"


def split_targets(fs, suite2p_ds, project=PROJECT):
    """Recordings, frame bounds and split datasets, in the order of `split_recordings`.

    Returns:
        list of (recording_id, first_frames (nplanes,), nframes, split Dataset).
    """
    datasets = flz.get_datasets_recursively(
        origin_id=suite2p_ds.origin_id,
        parent_type="recording",
        filter_parents={"recording_type": "two_photon"},
        dataset_type="scanimage",
        flexilims_session=fs,
        return_paths=True,
    )
    recording_ids = []
    for recording, paths in datasets.items():
        recording_ids.extend(itertools.repeat(recording, len(paths)))
    first_frames, last_frames = get_recording_frames(suite2p_ds)
    assert len(recording_ids) == len(first_frames)
    out = []
    for recording_id, first, last in zip(recording_ids, first_frames, last_frames):
        filt, excl = TRACES[project]
        split = flz.get_datasets(
            flexilims_session=fs,
            origin_id=recording_id,
            dataset_type="suite2p_traces",
            filter_datasets=filt,
            exclude_datasets=excl,
            allow_multiple=False,
            return_dataseries=False,
        )
        if split is None:
            raise FileNotFoundError(
                f"No suite2p_traces {filt} for recording {recording_id}"
            )
        out.append((recording_id, first, int(np.min(last - first)), split))
    return out


def rec_folder(split):
    """Recording folder name of a split dataset (split dataset names are not unique)."""
    return split.path_full.parent.name


def ops_update(suite2p_ds, iplane, tau):
    """Ops passed to the deconvolution: the new tau, and `baseline_method` for old ops.

    Ops of older suite2p versions (e.g. 0.11) only have `baseline`.
    """
    ops = np.load(
        suite2p_ds.path_full / f"plane{iplane}" / "ops.npy", allow_pickle=True
    ).tolist()
    new = {"tau": tau}
    if not ops.get("baseline_method"):
        new["baseline_method"] = ops["baseline"]
    return new


def deconvolve_plane(suite2p_ds, iplane, tau):
    """Same steps as `spike_deconvolution_suite2p(ast_neuropil=False)`, returned not saved."""
    plane = suite2p_ds.path_full / f"plane{iplane}"
    F = np.load(plane / "Fstandard.npy")
    ops = np.load(plane / "ops.npy", allow_pickle=True).tolist()
    ops.update(ops_update(suite2p_ds, iplane, tau))
    F = dcnv.preprocess(
        F=F,
        baseline=ops["baseline_method"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
    )
    return dcnv.oasis(F=F, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])


def check_only(suite2p_ds, targets, tau):
    """Recompute the spikes with `tau` and compare with the stored ones, without writing."""
    nplanes = int(float(suite2p_ds.extra_attributes["nplanes"]))
    ok = True
    for iplane in range(nplanes):
        plane = suite2p_ds.path_full / f"plane{iplane}"
        if np.load(plane / "F.npy").shape[1] == 0:
            continue
        spks = deconvolve_plane(suite2p_ds, iplane, tau)
        stored = np.load(plane / "spks.npy")
        same = stored.shape == spks.shape and np.array_equal(
            spks, stored, equal_nan=True
        )
        diff = (
            np.nanmax(np.abs(spks - stored)) if stored.shape == spks.shape else np.inf
        )
        corr = (
            np.nanmedian(
                [
                    np.corrcoef(a, b)[0, 1]
                    for a, b in zip(spks, stored)
                    if np.std(a) > 0 and np.std(b) > 0
                ]
            )
            if stored.shape == spks.shape
            else np.nan
        )
        print(
            f"plane {iplane} concatenated: identical {same}, max |diff| {diff:.3g}, median per-ROI corr {corr:.4f}"
        )
        ok &= same
        for recording_id, first, nframes, split in targets:
            part = spks[:, first[iplane] : first[iplane] + nframes]
            st = np.load(split.path_full / f"plane{iplane}" / "spks.npy")
            same = st.shape == part.shape and np.array_equal(part, st, equal_nan=True)
            print(f"  {rec_folder(split)}: identical {same}")
            ok &= same
    return ok


def main(
    session,
    tau,
    out_dir=None,
    backup_dir=None,
    compare=False,
    set_tau_attribute=False,
    project=PROJECT,
    check=False,
):
    fs = flz.get_flexilims_session(project)
    suite2p_ds, targets = concatenated_dataset(fs, session, project)
    nplanes = int(float(suite2p_ds.extra_attributes["nplanes"]))
    print(
        f"{session} {suite2p_ds.dataset_name} ({suite2p_ds.path_full}): {nplanes} plane(s), "
        f"{len(targets)} recordings, tau {tau} s; split indexing reproduces the stored split dff.npy"
    )
    if check:
        ok = check_only(suite2p_ds, targets, tau)
        print(f"RESULT {session} {'REPRODUCED' if ok else 'MISMATCH'}")
        return

    if out_dir is None and backup_dir is not None:
        backup = Path(backup_dir) / session
        if not backup.exists():
            for name, ds in [("concatenated", suite2p_ds)] + [
                (rec_folder(t[3]), t[3]) for t in targets
            ]:
                for iplane in range(nplanes):
                    src = ds.path_full / f"plane{iplane}" / "spks.npy"
                    if src.exists():
                        dst = backup / name / f"plane{iplane}" / "spks.npy"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
            (backup / "source.json").write_text(
                json.dumps(
                    {
                        "tau": suite2p_ds.extra_attributes.get("tau"),
                        "concatenated": str(suite2p_ds.path_full),
                        "split": {
                            rec_folder(t[3]): str(t[3].path_full) for t in targets
                        },
                    },
                    indent=1,
                )
            )
            print(f"backed up current spikes to {backup}")
        else:
            print(f"backup {backup} exists, kept")
    elif out_dir is None:
        raise ValueError("--backup-dir is required to overwrite the processed spikes")

    for iplane in range(nplanes):
        plane = suite2p_ds.path_full / f"plane{iplane}"
        if np.load(plane / "F.npy").shape[1] == 0:
            print(f"plane {iplane}: no ROIs, skipped")
            continue
        if out_dir is None:
            # the pipeline function itself, writing plane*/spks.npy
            spike_deconvolution_suite2p(
                suite2p_ds,
                iplane,
                ops=ops_update(suite2p_ds, iplane, tau),
                ast_neuropil=False,
            )
            spks = np.load(plane / "spks.npy")
        else:
            spks = deconvolve_plane(suite2p_ds, iplane, tau)
            dst = Path(out_dir) / "concatenated" / f"plane{iplane}" / "spks.npy"
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.save(dst, spks)
        if compare:
            stored = np.load(plane / "spks.npy")
            print(
                f"plane {iplane} concatenated vs stored: max |diff| {np.nanmax(np.abs(spks - stored)):.3g}, "
                f"identical {np.array_equal(spks, stored, equal_nan=True)}"
            )
        for recording_id, first, nframes, split in targets:
            part = spks[:, first[iplane] : first[iplane] + nframes]
            if out_dir is None:
                dst = split.path_full / f"plane{iplane}" / "spks.npy"
            else:
                dst = Path(out_dir) / rec_folder(split) / f"plane{iplane}" / "spks.npy"
                dst.parent.mkdir(parents=True, exist_ok=True)
            if compare:
                stored = np.load(split.path_full / f"plane{iplane}" / "spks.npy")
                assert (
                    stored.shape == part.shape
                ), f"{rec_folder(split)}: {stored.shape} vs {part.shape}"
                same = np.array_equal(part, stored, equal_nan=True)
                print(
                    f"  {rec_folder(split)}: max |diff| {np.nanmax(np.abs(part - stored)):.3g}, identical {same}"
                )
            np.save(dst, part)
    if out_dir is not None:
        (Path(out_dir) / "paths.json").write_text(
            json.dumps(
                {rec_folder(t[3]): str(t[3].path_full) for t in targets}, indent=1
            )
        )
    if out_dir is None:
        print(f"overwrote spks.npy of {1 + len(targets)} datasets")
        if set_tau_attribute:
            for ds in [suite2p_ds] + [t[3] for t in targets]:
                ds.extra_attributes["tau"] = tau
                ds.update_flexilims(mode="update")
            print(f"set tau = {tau} on flexilims")
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument(
        "--out-dir",
        default=None,
        help="write here instead of overwriting the processed spikes",
    )
    p.add_argument(
        "--backup-dir",
        default=None,
        help="copy the current spikes here before overwriting",
    )
    p.add_argument(
        "--compare", action="store_true", help="compare with the stored spks.npy"
    )
    p.add_argument(
        "--set-tau-attribute", action="store_true", help="record tau on flexilims"
    )
    p.add_argument("--project", default=PROJECT, choices=sorted(TRACES))
    p.add_argument(
        "--check-only",
        action="store_true",
        help="compare with the stored spikes, write nothing",
    )
    a = p.parse_args()
    main(
        a.session,
        a.tau,
        a.out_dir,
        a.backup_dir,
        a.compare,
        a.set_tau_attribute,
        a.project,
        a.check_only,
    )
