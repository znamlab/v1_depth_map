"""Copy the data needed to run `v1_depth_map/figures/figsupp_multidays.ipynb`.

The notebook reads from the `colasa_3d-vision_revisions` project: per-day
`neurons_df.pickle` for every mouse/day sphere-tube session, the manual ROICat
cross-day tracking files, and everything `spheres.sync_all_recordings` needs to
rebuild `trials_df` for the `SpheresPermTubeReward` recordings (visstim/harp raw
files, the cached harpmessage/monitor_frames pickles, and the annotated
suite2p_traces dff/spks arrays).

Destination layout matches a flexiznam config pointing at BlackPasspo, e.g.::

    colasa_3d-vision_revisions:
        processed: /Volumes/BlackPasspo/v1_depth_map/processed
        raw: /Volumes/BlackPasspo/v1_depth_map/raw
"""

import argparse
import shutil
import time
from pathlib import Path

import flexiznam as flz

from v1_depth_map.revisions.revision_sessions import sessions as rev_sessions

# Suffixes of raw imaging data we never need for this notebook (huge, and
# sync_all_recordings only reads the pre-computed suite2p traces / caches).
HEAVY_SUFFIXES = {".tif", ".tiff", ".mp4", ".avi"}

RECORDING_DATASET_TYPES = ["visstim", "harp", "harp_npz", "monitor_frames"]


def robust_copy2(src, target, retries=5, delay=10, skip_existing=False):
    src = Path(src)
    target = Path(target)
    if src.name.startswith("._"):
        return
    if skip_existing and target.exists():
        return
    for i in range(retries):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, target)
            except OSError:
                shutil.copy(src, target)
            return
        except (OSError, TimeoutError) as e:
            if i < retries - 1:
                print(
                    f"  [RETRY {i + 1}/{retries}] Failed copying file {src.name} "
                    f"due to {e}. Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                raise


def _ignore_heavy(dirpath, filenames):
    ignored = []
    for fname in filenames:
        if fname.startswith("._"):
            ignored.append(fname)
        elif Path(fname).suffix.lower() in HEAVY_SUFFIXES:
            ignored.append(fname)
    return ignored


def robust_copytree(src, target, retries=5, delay=10, skip_existing=False):
    """Always walk into `src`, even if `target` already exists; only individual
    files that already exist in `target` are skipped (when `skip_existing`)."""

    def _copy_file(s, d):
        robust_copy2(s, d, retries=1, skip_existing=skip_existing)

    for i in range(retries):
        try:
            shutil.copytree(
                src,
                target,
                dirs_exist_ok=True,
                ignore=_ignore_heavy,
                copy_function=_copy_file,
            )
            return
        except (OSError, TimeoutError) as e:
            if i < retries - 1:
                print(
                    f"  [RETRY {i + 1}/{retries}] Failed copying directory {src.name} "
                    f"due to {e}. Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                raise


def copy_path(src_path, target_path, skip_existing=False):
    if src_path.is_file():
        if src_path.suffix.lower() in HEAVY_SUFFIXES:
            return
        robust_copy2(src_path, target_path, skip_existing=skip_existing)
    elif src_path.is_dir():
        robust_copytree(src_path, target_path, skip_existing=skip_existing)


def _normalise_ds_list(ds_list, flm_sess):
    """flz.get_datasets can return a Dataset, a Series, a DataFrame, a list or None."""
    import pandas as pd

    if ds_list is None:
        return []
    if isinstance(ds_list, pd.DataFrame):
        return [
            flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess)
            for _, ds in ds_list.iterrows()
        ]
    if isinstance(ds_list, pd.Series):
        return [
            flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess)
            for ds in ds_list
        ]
    if isinstance(ds_list, list):
        return ds_list
    return [ds_list]


def get_neurons_df_paths(src_processed_project_root, mouse, sess):
    """Paths for the `neurons_df.pickle` (+ sibling parquet files) read directly
    off disk by the notebook, at `{src_processed_project_root}/{mouse}/{sess}/`.
    `src_processed_project_root` must already point at the real network location
    (see --src-processed-root), not a possibly-redirected flexiznam data_root."""
    session_dir = src_processed_project_root / mouse / sess
    paths = set()
    neurons_pickle = session_dir / "neurons_df.pickle"
    if neurons_pickle.exists():
        paths.add(neurons_pickle)
    for p in session_dir.glob("*.parquet"):
        paths.add(p)
    return paths


def get_roicat_paths(src_processed_project_root, mouse):
    """Manual-click cross-day tracking files used to build `tracking_uid_manual`."""
    paths = set()
    roicat_dir = src_processed_project_root / mouse / "ROICat_spheretubes"
    for fname in [f"{mouse}_roicat.tracking.params_used.json", "matches.json"]:
        p = roicat_dir / fname
        if p.exists():
            paths.add(p)
    return paths


def get_suite2p_traces_paths(flm_sess, recording_name):
    """suite2p_traces dataset matching the filter used by `spheres.sync_all_recordings`
    in the notebook (anatomical_only:3, annotated:True), falling back to
    annotated:True alone if that stricter filter finds nothing.

    Returns raw `path_full` values as resolved by flexiznam (not existence-checked:
    the local flexiznam config may already redirect this project's data_root to the
    copy destination, so `.exists()` here can't be trusted — see
    `resolve_src_and_target`)."""
    ds_list = []
    for filt in ({"anatomical_only": 3, "annotated": True}, {"annotated": True}):
        ds_list = _normalise_ds_list(
            flz.get_datasets(
                origin_name=recording_name,
                dataset_type="suite2p_traces",
                filter_datasets=filt,
                flexilims_session=flm_sess,
                allow_multiple=True,
            ),
            flm_sess,
        )
        if ds_list:
            break

    return {ds.path_full for ds in ds_list if hasattr(ds, "path_full") and ds.path_full}


def get_recording_paths(flm_sess, session_name, protocols):
    """Everything `spheres.sync_all_recordings` needs for the recordings of
    `session_name` matching one of `protocols`: visstim/harp raw datasets, the
    cached harpmessage/monitor_frames pickles, and the annotated suite2p_traces.

    Returns raw `path_full` values (see `get_suite2p_traces_paths` docstring on why
    these are not existence-checked here)."""
    paths = set()
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flm_sess
    )
    if exp_session is None:
        return paths

    recordings = flz.get_children(
        parent_id=exp_session.id,
        flexilims_session=flm_sess,
        children_datatype="recording",
    )
    if recordings is None or recordings.empty:
        return paths

    valid_recordings = recordings[recordings.protocol.str.contains("|".join(protocols))]
    for rec_name, rec in valid_recordings.iterrows():
        for dataset_type in RECORDING_DATASET_TYPES:
            ds_list = _normalise_ds_list(
                flz.get_datasets(
                    origin_name=rec_name,
                    dataset_type=dataset_type,
                    flexilims_session=flm_sess,
                    allow_multiple=True,
                ),
                flm_sess,
            )
            for ds in ds_list:
                if hasattr(ds, "path_full") and ds.path_full:
                    paths.add(ds.path_full)

        paths |= get_suite2p_traces_paths(flm_sess, rec_name)

    return paths


def resolve_src_and_target(
    p,
    processed_root_flz,
    raw_root_flz,
    src_processed_root,
    src_raw_root,
    dest_processed,
    dest_raw,
):
    """`p` is a path as resolved by flexiznam (`processed_root_flz`/`raw_root_flz`),
    which may already be redirected to `dest` by the local flexiznam config (e.g. a
    config already pointing at BlackPasspo). Re-root it onto the real network
    location (`src_processed_root`/`src_raw_root`) to copy from, and onto `dest` to
    copy to."""
    try:
        rel = p.relative_to(processed_root_flz)
        return src_processed_root / rel, dest_processed / rel
    except ValueError:
        pass
    try:
        rel = p.relative_to(raw_root_flz)
        return src_raw_root / rel, dest_raw / rel
    except ValueError:
        pass
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Copy data needed by figsupp_multidays.ipynb to a local drive."
    )
    parser.add_argument(
        "dest",
        type=str,
        nargs="?",
        default="/Volumes/BlackPasspo/v1_depth_map",
        help="Destination root directory (will contain processed/ and raw/ subfolders).",
    )
    parser.add_argument(
        "--src-project",
        type=str,
        default="colasa_3d-vision_revisions",
        help="Source project ID in flexilims.",
    )
    parser.add_argument(
        "--src-processed-root",
        type=str,
        default="/Volumes/lab-znamenskiyp/home/shared/projects",
        help=(
            "Real network location to copy processed data from. Needed because "
            "flexiznam config may already redirect this project's data_root to "
            "`dest` (e.g. a config already pointing at BlackPasspo)."
        ),
    )
    parser.add_argument(
        "--src-raw-root",
        type=str,
        default="/Volumes/proj-znamenp-3dvision/raw",
        help="Real network location to copy raw data from (see --src-processed-root).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the paths that would be copied without copying.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip individual files that already exist in the destination. "
            "Directories are always walked into (never skipped wholesale), so "
            "an interrupted directory copy can be resumed."
        ),
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest_processed = dest / "processed"
    dest_raw = dest / "raw"

    src_processed_root = Path(args.src_processed_root)
    src_raw_root = Path(args.src_raw_root)
    src_processed_project_root = src_processed_root / args.src_project

    print(f"Connecting to flexilims (project: {args.src_project})...")
    flm_sess = flz.get_flexilims_session(project_id=args.src_project)

    processed_root_flz = flz.get_data_root("processed", flexilims_session=flm_sess)
    raw_root_flz = flz.get_data_root("raw", flexilims_session=flm_sess)

    sphere_sessions = [(k, v) for k, v in rev_sessions.items() if "sphere" in v]
    mice = sorted(set(sess_name.split("_")[0] for sess_name, _ in sphere_sessions))
    print(f"Found {len(sphere_sessions)} sphere sessions across {len(mice)} mice.")

    # (real source path, destination path) pairs.
    items = {}

    from tqdm import tqdm

    for sess_name, _ in tqdm(
        sphere_sessions, desc="Discovering sessions", unit="session"
    ):
        mouse, sess = sess_name.split("_")

        for p in get_neurons_df_paths(src_processed_project_root, mouse, sess):
            rel = p.relative_to(src_processed_root)
            items[str(p)] = (p, dest_processed / rel)

        for p in get_recording_paths(flm_sess, sess_name, ["SpheresPermTubeReward"]):
            src_p, target_p = resolve_src_and_target(
                p,
                processed_root_flz,
                raw_root_flz,
                src_processed_root,
                src_raw_root,
                dest_processed,
                dest_raw,
            )
            if src_p is None:
                print(f"WARNING: {p} is outside raw/processed roots. Skipping.")
                continue
            items[str(src_p)] = (src_p, target_p)

    for mouse in mice:
        for p in get_roicat_paths(src_processed_project_root, mouse):
            rel = p.relative_to(src_processed_root)
            items[str(p)] = (p, dest_processed / rel)

    print(f"Discovered {len(items)} paths to copy.")

    if args.dry_run:
        print("\n--- Dry Run: Discovery Analysis ---")
        for src_p, target_p in sorted(items.values(), key=lambda x: str(x[0])):
            status = " [FOUND]" if src_p.exists() else " [MISSING]"
            print(f"{status} {src_p} -> {target_p}")
        return

    dest_processed.mkdir(parents=True, exist_ok=True)
    dest_raw.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(
        sorted(items.values(), key=lambda x: str(x[0])),
        desc="Syncing files",
        unit="item",
    )
    for src_p, target_p in pbar:
        pbar.set_description(f"Syncing {src_p.name}")
        if not src_p.exists():
            pbar.write(f"WARNING: Source {src_p} does not exist. Skipping.")
            continue

        copy_path(src_p, target_p, skip_existing=args.skip_existing)

    print("Data sync successfully completed!")


if __name__ == "__main__":
    main()
