import argparse
import shutil
from pathlib import Path
import flexiznam as flz
import pandas as pd
import time

def robust_copy2(src, target, retries=5, delay=10):
    if src.name.startswith("._"):
        return False
    for i in range(retries):
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            return True
        except (OSError, TimeoutError) as e:
            if i < retries - 1:
                print(f"  [RETRY {i+1}/{retries}] Failed copying file {src.name} due to {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

def robust_copytree(src, target, retries=5, delay=10, ignore=None):
    for i in range(retries):
        try:
            shutil.copytree(src, target, dirs_exist_ok=True, ignore=ignore)
            return True
        except (OSError, TimeoutError) as e:
            if i < retries - 1:
                print(f"  [RETRY {i+1}/{retries}] Failed copying directory {src.name} due to {e}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise

def copy_processed_session(flm_sess, session_name, dest_processed_dir):
    """Copy all processed session datasets needed."""
    paths_to_copy = set()
    neurons_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="neurons_df",
        flexilims_session=flm_sess,
        allow_multiple=False,
    )
    if neurons_ds is None:
        return paths_to_copy
        
    paths_to_copy.add(neurons_ds.path_full)
    
    # 1. simulated responses
    for p in neurons_ds.path_full.parent.glob("*.parquet"):
        paths_to_copy.add(p)
            
    # 2. suite2p iscell, ops, stat
    suite2p_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={'annotated': True},
        flexilims_session=flm_sess,
        allow_multiple=False,
    )
    if suite2p_ds is not None:
        for plane_dir in sorted(suite2p_ds.path_full.glob("plane*")):
            if plane_dir.is_dir():
                for fname in ["iscell.npy", "ops.npy", "stat.npy"]:
                    fpath = plane_dir / fname
                    if fpath.exists():
                        paths_to_copy.add(fpath)
                        
    # 3. monitor_frames_df
    monitor_frames_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="monitor_frames_df",
        flexilims_session=flm_sess,
        allow_multiple=True,
    )
    if monitor_frames_ds is not None:
        if isinstance(monitor_frames_ds, pd.DataFrame):
            for _, ds in monitor_frames_ds.iterrows():
                if hasattr(ds, "path_full") and ds.path_full.exists():
                    paths_to_copy.add(ds.path_full)
        else:
            for ds in monitor_frames_ds:
                if hasattr(ds, "path_full") and ds.path_full.exists():
                    paths_to_copy.add(ds.path_full)

    return paths_to_copy

def copy_raw_session(flm_sess, session_name, dest_raw_dir, protocols):
    """Copy raw metadata files and target video files if required."""
    paths_to_copy = set()
    exp_session = flz.get_entity(datatype="session", name=session_name, flexilims_session=flm_sess)
    if exp_session is None: 
        return paths_to_copy
        
    recordings = flz.get_children(
        parent_id=exp_session.id,
        flexilims_session=flm_sess,
        children_datatype="recording",
    )
    
    valid_recordings = recordings[recordings.protocol.str.contains("|".join(protocols))]
    for _, rec in valid_recordings.iterrows():
        datasets = flz.get_children(parent_id=rec.id, flexilims_session=flm_sess, children_datatype="dataset")
        for _, ds in datasets.iterrows():
            dataset_obj = flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess)
            if hasattr(dataset_obj, "path_full") and dataset_obj.path_full.exists():
                paths_to_copy.add(dataset_obj.path_full)
                
    return paths_to_copy

def main():
    parser = argparse.ArgumentParser(description="Copy raw behavioral and processed session traces for V1 depth map figures to BlackPasspo.")
    parser.add_argument("dest", type=str, default="/Volumes/BlackPasspo/v1_depth_map", nargs="?", help="Destination root directory for figures project.")
    parser.add_argument("--revisions-dest", type=str, default=None, help="Destination root directory for treadmill/revisions project. Defaults to the same as dest.")
    parser.add_argument("--src-project", type=str, default="hey2_3d-vision_foodres_20220101", help="Flexilims project name.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the paths that would be copied without copying.")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest_processed = dest / "processed"
    dest_raw = dest / "raw"
    
    dest_rev = Path(args.revisions_dest) if args.revisions_dest else dest
    dest_rev_processed = dest_rev / "processed"
    dest_rev_raw = dest_rev / "raw"
    
    if not args.dry_run:
        dest_processed.mkdir(parents=True, exist_ok=True)
        dest_raw.mkdir(parents=True, exist_ok=True)
        if dest_rev != dest:
            dest_rev_processed.mkdir(parents=True, exist_ok=True)
            dest_rev_raw.mkdir(parents=True, exist_ok=True)

    print("Connecting to flexilims for figures project...")
    flm_sess = flz.get_flexilims_session(project_id=args.src_project)
    processed_root = flz.get_data_root("processed", flexilims_session=flm_sess)
    raw_root = flz.get_data_root("raw", flexilims_session=flm_sess)

    all_paths_to_copy = []

    # 1. Database JSON Copy
    db_src = Path("/Volumes/lab-znamenskiyp/home/shared/projects/offline_database.json")
    if db_src.exists():
        all_paths_to_copy.append((db_src, dest_processed / "offline_database.json", "file"))
        if dest_rev != dest:
            all_paths_to_copy.append((db_src, dest_rev_processed / "offline_database.json", "file"))

    else:
        print(f"WARNING: Database file {db_src} not found!")

    # 2. Precomputed eye tracking analysis data
    src_eye = processed_root / "Analysis" / "eye_tracking"
    if src_eye.exists():
        all_paths_to_copy.append((src_eye, dest_processed / "hey2_3d-vision_foodres_20220101" / "Analysis" / "eye_tracking", "dir"))

    # V1 mice sessions definition
    v1_mice = ["PZAH6.4b", "PZAG3.4f", "PZAH8.2h", "PZAH8.2i", "PZAH8.2f", "PZAH10.2d", "PZAH10.2f"]
    all_sessions = flz.get_entities(datatype="session", flexilims_session=flm_sess)
    
    print("Discovering figures project session datasets...")
    for idx, session in all_sessions.iterrows():
        sess_name = session["name"]
        mouse_name = sess_name.split("_")[0]
        if mouse_name not in v1_mice:
            continue

        # A. Processed copy
        processed_paths = copy_processed_session(flm_sess, sess_name, dest_processed)
        if processed_paths:
            for p in processed_paths:
                rel = p.relative_to(processed_root)
                target = dest_processed / rel
                all_paths_to_copy.append((p, target, "file"))

        # B. Raw metadata & video copy
        raw_paths = copy_raw_session(flm_sess, sess_name, dest_raw, ["SpheresPermTubeReward", "SizeControl"])
        for p in raw_paths:
            p = Path(p)
            try:
                rel = p.relative_to(raw_root)
                target = dest_raw / rel
            except ValueError:
                try:
                    rel = p.relative_to(processed_root)
                    target = dest_processed / rel
                except ValueError:
                    print(f"WARNING: Path {p} is outside raw and processed roots. Skipping.")
                    continue

            if p.is_file():
                # Avoid copying heavy videos unless it's the exact eye tracking calibration session
                if "video_file" in p.name or p.suffix in [".mp4", ".avi", ".tif"]:
                    if sess_name != "PZAG3.4f_S20220421": 
                        continue
                all_paths_to_copy.append((p, target, "file"))
            elif p.is_dir():
                all_paths_to_copy.append((p, target, "custom_dir"))

    # Revisions project discovery
    print("Connecting to flexilims for revisions project...")
    try:
        from revision_sessions import sessions as rev_sessions
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from revision_sessions import sessions as rev_sessions

    flm_rev = flz.get_flexilims_session(project_id="colasa_3d-vision_revisions")
    processed_root_rev = flz.get_data_root("processed", flexilims_session=flm_rev)
    raw_root_rev = flz.get_data_root("raw", flexilims_session=flm_rev)

    print("Discovering revisions project session datasets...")
    for sess_name, protocol in rev_sessions.items():
        # A. Processed copy
        processed_paths = copy_processed_session(flm_rev, sess_name, dest_rev_processed)
        if processed_paths:
            for p in processed_paths:
                rel = p.relative_to(processed_root_rev)
                target = dest_rev_processed / rel
                all_paths_to_copy.append((p, target, "file"))

        # B. Raw metadata copy (no videos/tiffs needed for revision notebooks)
        raw_paths = copy_raw_session(flm_rev, sess_name, dest_rev_raw, ["SpheresTubeMotor", "SpheresPermTubeReward"])
        for p in raw_paths:
            p = Path(p)
            try:
                rel = p.relative_to(raw_root_rev)
                target = dest_rev_raw / rel
            except ValueError:
                try:
                    rel = p.relative_to(processed_root_rev)
                    target = dest_rev_processed / rel
                except ValueError:
                    print(f"WARNING: Path {p} is outside raw and processed roots. Skipping.")
                    continue

            if p.is_file():
                if "video_file" in p.name or p.suffix in [".mp4", ".avi", ".tif"]:
                    continue
                all_paths_to_copy.append((p, target, "file"))
            elif p.is_dir():
                all_paths_to_copy.append((p, target, "custom_dir"))

    print(f"Total paths discovered: {len(all_paths_to_copy)}")


    if args.dry_run:
        print("\n--- Dry Run: Discovery Analysis ---")
        for src, target, mode in sorted(all_paths_to_copy):
            status = " [FOUND]" if src.exists() else " [MISSING]"
            print(f"{status} {src} -> {target} ({mode})")
        return

    # Copy files
    print(f"Syncing files to {dest}...")
    for src, target, mode in sorted(all_paths_to_copy):
        if not src.exists():
            print(f"WARNING: Source {src} does not exist. Skipping.")
            continue

        if mode == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and target.exists():
                print(f"Skipping file {src.name} (exists)")
                continue
            robust_copy2(src, target)
            print(f"Copied file {src.name}")

        elif mode == "dir":
            if args.skip_existing and target.exists():
                print(f"Skipping dir {src.name} (exists)")
                continue
            robust_copytree(src, target)
            print(f"Copied directory {src.name}")

        elif mode == "custom_dir":
            target.mkdir(parents=True, exist_ok=True)
            
            # Determine session name from source path (e.g. .../mouse/session_date/recording)
            try:
                mouse_name = src.parent.parent.name
                sess_date = src.parent.name
                sess_name = f"{mouse_name}_{sess_date}"
            except Exception:
                sess_name = ""

            # Define which files are actually needed for figures dynamic loading
            if sess_name == "PZAG3.4f_S20220421":
                allowed_filenames = {
                    "FrameLog.csv", "NewParams.csv", "ParamLog.csv", "harpmessage.npz",
                    "left_eye_camera_timestamps.csv", "right_eye_camera_timestamps.csv",
                    "face_camera_timestamps.csv", "butt_camera_timestamps.csv"
                }
            else:
                allowed_filenames = {
                    "FrameLog.csv", "NewParams.csv", "ParamLog.csv", "harpmessage.npz"
                }

            copied_count = 0
            if src.exists():
                for p_file in src.iterdir():
                    if p_file.is_file():
                        fname = p_file.name
                        if fname.startswith("._"):
                            continue
                        is_allowed = (fname in allowed_filenames) or fname.startswith("NewParams_") or fname.startswith("ParamLog_")
                        if is_allowed:
                            target_file = target / fname
                            if args.skip_existing and target_file.exists():
                                continue
                            robust_copy2(p_file, target_file)
                            copied_count += 1
                        
            # Sync suite2p trace subfolders
            for sub in ["suite2p_traces_0", "suite2p_traces_annotated_0"]:
                src_sub = src / sub
                target_sub = target / sub
                if src_sub.exists():
                    if args.skip_existing and target_sub.exists():
                        continue
                    robust_copytree(src_sub, target_sub, ignore=shutil.ignore_patterns("*.tif", "*.mp4", "._*"))
            print(f"Synced metadata and traces for {src.name} ({copied_count} files)")

    print("Data sync successfully completed!")

if __name__ == "__main__":
    main()
