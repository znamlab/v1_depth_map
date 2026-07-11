import argparse
import shutil
from pathlib import Path
import flexiznam as flz


def get_session_datasets(flm_sess, session_name, protocol_base="SpheresTubeMotor"):
    """Discover all datasets for a given session that are required by the notebook."""
    paths_to_copy = set()

    # 1. neurons_df
    neurons_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="neurons_df",
        flexilims_session=flm_sess,
        allow_multiple=False,
    )
    if neurons_ds is None:
        return paths_to_copy

    paths_to_copy.add(neurons_ds.path_full)

    # 2. Simulated responses
    # Format used in notebook: simulated_responses_fit_{which}_{TDECAY}_{TRISE}_{CIRC}.parquet
    # We copy all parquet files in the session folder to be safe.
    for p in neurons_ds.path_full.parent.glob("*.parquet"):
        paths_to_copy.add(p)

    # 3. suite2p iscell
    suite2p_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={"annotated": True},
        flexilims_session=flm_sess,
        allow_multiple=False,
    )
    if suite2p_ds is not None:
        # We need iscell.npy, ops.npy, stat.npy for all planes (usually just plane0)
        # s2p_io.load_is_cell might look in plane0 or the root depending on version
        for plane_dir in sorted(suite2p_ds.path_full.glob("plane*")):
            if plane_dir.is_dir():
                for fname in ["iscell.npy", "ops.npy", "stat.npy"]:
                    fpath = plane_dir / fname
                    if fpath.exists():
                        paths_to_copy.add(fpath)

    return paths_to_copy


def get_example_session_datasets(flm_sess, session_name):
    """Discover the raw datasets of the example session necessary for pipeline_utils.load_treadmill_and_sphere_datasets"""
    paths_to_copy = set()
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flm_sess
    )

    if exp_session is None:
        raise ValueError(f"Session {session_name} not found.")

    # It loads SpheresTubeMotor and SpheresPermTubeReward recordings
    recordings = flz.get_children(
        parent_id=exp_session.id,
        flexilims_session=flm_sess,
        children_datatype="recording",
    )

    valid_recordings = recordings[
        recordings.protocol.str.contains("SpheresTubeMotor|SpheresPermTubeReward")
    ]

    # Get all datasets for these recordings (photodiode, suite2p, visstim, harp etc.)
    for _, rec in valid_recordings.iterrows():
        datasets = flz.get_children(
            parent_id=rec.id, flexilims_session=flm_sess, children_datatype="dataset"
        )
        for _, ds in datasets.iterrows():
            dataset_obj = flz.Dataset.from_flexilims(
                id=ds.id, flexilims_session=flm_sess
            )
            if hasattr(dataset_obj, "path_full") and dataset_obj.path_full.exists():
                paths_to_copy.add(dataset_obj.path_full)

    return paths_to_copy


def main():
    parser = argparse.ArgumentParser(
        description="Copy data used by treadmill.ipynb to a specific destination."
    )
    parser.add_argument("dest", type=str, help="Destination directory path.")
    parser.add_argument(
        "--src-project",
        type=str,
        default="colasa_3d-vision_revisions",
        help="Source project ID in flexilims.",
    )
    parser.add_argument(
        "--src-raw-root",
        type=str,
        help="Override the source raw data root (e.g. if a volume is mounted at a non-standard path).",
    )
    parser.add_argument(
        "--src-processed-root",
        type=str,
        help="Override the source processed data root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the paths that would be copied without copying.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files and directories that already exist in the destination.",
    )
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to flexilims (project: {args.src_project})...")
    flm_sess = flz.get_flexilims_session(project_id=args.src_project)

    mice = flz.get_entities(datatype="mouse", flexilims_session=flm_sess)
    all_sessions = flz.get_entities(datatype="session", flexilims_session=flm_sess)

    protocol_base = "SpheresTubeMotor"
    treadmill_sessions = {}

    print("Finding valid treadmill sessions...")
    for mouse_name, mouse_data in mice.iterrows():
        sessions = all_sessions[all_sessions.origin_id == mouse_data.id]
        for session_name, sess_data in sessions.iterrows():
            recordings = flz.get_children(
                parent_id=sess_data.id,
                flexilims_session=flm_sess,
                children_datatype="recording",
            )
            if not len(recordings):
                continue
            if protocol_base in recordings.protocol.values:
                treadmill_sessions[session_name] = recordings

    print(f"Found {len(treadmill_sessions)} sessions with {protocol_base} protocol.")

    all_paths_to_copy = set()

    for session_name in treadmill_sessions.keys():
        paths = get_session_datasets(flm_sess, session_name)
        all_paths_to_copy.update(paths)

    # The example session is hardcoded
    example_session = "PZAG17.3a_S20250402"
    print(f"Finding datasets for example session {example_session}...")
    example_paths = get_example_session_datasets(flm_sess, example_session)
    all_paths_to_copy.update(example_paths)

    print(f"Discovered {len(all_paths_to_copy)} paths to copy.")

    # roots for relative path calculation and override logic
    processed_root_flz = flz.get_data_root("processed", flexilims_session=flm_sess)
    raw_root_flz = flz.get_data_root("raw", flexilims_session=flm_sess)

    src_processed_root = (
        Path(args.src_processed_root) if args.src_processed_root else processed_root_flz
    )
    src_raw_root = Path(args.src_raw_root) if args.src_raw_root else raw_root_flz

    if args.dry_run:
        print("\n--- Dry Run: Discovery Analysis ---")
        for p in sorted(all_paths_to_copy):
            p = Path(p)
            # Apply override if specified
            if args.src_processed_root:
                try:
                    p = src_processed_root / p.relative_to(processed_root_flz)
                except ValueError:
                    pass
            if args.src_raw_root:
                try:
                    p = src_raw_root / p.relative_to(raw_root_flz)
                except ValueError:
                    pass

            status = " [FOUND]" if p.exists() else " [MISSING]"
            print(f"{status} {p}")
            if not p.exists():
                parts = p.parts
                if len(parts) > 2 and parts[1] == "Volumes":
                    print(f"      --> CHECK: Is the volume '{parts[2]}' mounted?")
        return

    # Copy files
    print(f"Copying files to {dest}...")

    for p in sorted(all_paths_to_copy):
        p = Path(p)
        # Apply override if specified
        if args.src_processed_root:
            try:
                p = src_processed_root / p.relative_to(processed_root_flz)
            except ValueError:
                pass
        if args.src_raw_root:
            try:
                p = src_raw_root / p.relative_to(raw_root_flz)
            except ValueError:
                pass

        if not p.exists():
            print(f"CRITICAL WARNING: Source path {p} does not exist. Skipping.")
            parts = p.parts
            if len(parts) > 2 and parts[1] == "Volumes":
                print(f"  --> Is the volume '{parts[2]}' mounted?")
            continue

        # Try to find a good relative path for target
        try:
            rel_path = p.relative_to(src_processed_root)
        except ValueError:
            try:
                rel_path = p.relative_to(src_raw_root)
            except ValueError:
                rel_path = Path(p.name)  # Just use item name as fallback

        target_path = dest / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Avoid SameFileError and apply skip-existing logic
        if target_path.exists():
            if target_path.resolve() == p.resolve():
                print(
                    f"Skipping {p.name} (source and destination are the exact same path)"
                )
                continue
            if args.skip_existing and p.is_file():
                print(f"Skipping {p.name} (already exists)")
                continue

        if p.is_file():
            shutil.copy2(p, target_path)
            print(f"Copied file {p.name}")
        elif p.is_dir():
            # Optimize: Explicitly target metadata files to avoid listing thousands of images
            target_path.mkdir(parents=True, exist_ok=True)
            target_files = [
                "FrameLog.csv",
                "AllParams.csv",
                "NewParams.csv",
                "RewardLog.csv",
                "RotaryEncoder.csv",
                "harpmessage.npz",
                "monitor_frames_df.pickle",
                "harpmessage.bin",
                "butt_camera_timestamps.csv",
                "face_camera_timestamps.csv",
                "left_eye_camera_timestamps.csv",
                "right_eye_camera_timestamps.csv",
            ]

            print(f"Syncing metadata from {p.name}...")
            copied_count = 0
            for fname in target_files:
                src_file = p / fname
                if src_file.exists():
                    shutil.copy2(src_file, target_path / fname)
                    copied_count += 1
                    if fname == "FrameLog.csv":
                        print("  [SUCCESS] Found and copied FrameLog.csv")

            # Special case for suite2p subfolders if they exist
            for sub in ["suite2p_traces_0", "suite2p_traces_annotated_0"]:
                src_sub = p / sub
                if src_sub.exists():
                    # Keep using copytree for suite2p as they are smaller/structured
                    shutil.copytree(
                        src_sub,
                        target_path / sub,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.tif", "*.mp4", "._*"),
                    )

            print(f"  Copied {copied_count} metadata files from {p.name}")

    print("Done!")


if __name__ == "__main__":
    main()
