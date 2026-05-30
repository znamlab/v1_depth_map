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

def suite2p_traces_ignore(dirpath, filenames):
    ignore_list = []
    for fname in filenames:
        if fname.endswith(".npy"):
            if fname not in ["dff.npy", "dff_ast.npy", "iscell.npy", "ops.npy", "stat.npy", "spks.npy"]:
                ignore_list.append(fname)
        elif fname.endswith(".tif") or fname.endswith(".mp4") or fname.startswith("._"):
            ignore_list.append(fname)
    return ignore_list

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
        return paths_to_copy, None

    processed_root_flz = flz.get_data_root("processed", flexilims_session=flm_sess)
    src_processed_root = Path("/Volumes/lab-znamenskiyp/home/shared/projects")

    # Resolve neurons_ds.path_full to server path
    try:
        rel = neurons_ds.path_full.relative_to(processed_root_flz)
        src_neurons_path = src_processed_root / rel
    except ValueError:
        src_neurons_path = neurons_ds.path_full
        
    paths_to_copy.add(src_neurons_path)
    
    # 1. simulated responses
    for p in src_neurons_path.parent.glob("*.parquet"):
        paths_to_copy.add(p)

    # 1b. per-session decoder results (read by depth_decoder_stats.concatenate_all_decoder_results)
    decoder_pickle = src_neurons_path.parent / "decoder_results.pickle"
    if decoder_pickle.exists():
        paths_to_copy.add(decoder_pickle)
            
    # 2. suite2p stat/iscell/ops/dff — copy from the anatomical_only:3 ROI set
    # The notebook loads stat/iscell via filter_datasets={"anatomical_only": 3, "ast_neuropil":False} (no
    # ast_neuropil constraint), so we must match that here.
    # First try: figures project filter (anatomical_only:3), any ast_neuropil value.
    # Second try: revisions project filter (annotated:True), any ast_neuropil value.
    def _normalise_ds_list(ds_list):
        if isinstance(ds_list, pd.DataFrame):
            return [flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess) for _, ds in ds_list.iterrows()]
        elif isinstance(ds_list, pd.Series):
            return [flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess) for ds in ds_list]
        elif ds_list is None:
            return []
        elif not isinstance(ds_list, list):
            return [ds_list]
        return ds_list

    suite2p_ds_list = _normalise_ds_list(flz.get_datasets(
        origin_name=session_name,
        dataset_type="suite2p_rois",
        filter_datasets={'anatomical_only': 3},
        flexilims_session=flm_sess,
        allow_multiple=True,
    ))

    if not suite2p_ds_list:
        suite2p_ds_list = _normalise_ds_list(flz.get_datasets(
            origin_name=session_name,
            dataset_type="suite2p_rois",
            filter_datasets={'annotated': True},
            flexilims_session=flm_sess,
            allow_multiple=True,
        ))

    # Prefer the ast_neuropil:False dataset; otherwise take the last one.
    suite2p_ds = None
    if suite2p_ds_list:
        no_ast = [ds for ds in suite2p_ds_list if not ds._extra_attributes.get('ast_neuropil', True)]
        annotated = [ds for ds in suite2p_ds_list if ds._extra_attributes.get('annotated', False)]
        if no_ast:
            suite2p_ds = no_ast[-1]
        elif annotated:
            suite2p_ds = annotated[-1]
        else:
            suite2p_ds = suite2p_ds_list[-1]

    suite2p_suffix = None
    if suite2p_ds is not None:
        # Extract suffix (e.g. "1" from "suite2p_rois_1")
        name = suite2p_ds.path_full.name
        if "_" in name:
            suite2p_suffix = name.split("_")[-1]

        # Resolve suite2p_ds.path_full to server path
        try:
            rel = suite2p_ds.path_full.relative_to(processed_root_flz)
            src_suite2p_path = src_processed_root / rel
        except ValueError:
            src_suite2p_path = suite2p_ds.path_full

        # Copy ROI definition files (stat/iscell/ops) — ast_neuropil is irrelevant here.
        # Trace files (dff/spks) are handled exclusively by step 3 below,
        # from suite2p_traces datasets with ast_neuropil:False at recording level.
        for plane_dir in sorted(src_suite2p_path.glob("plane*")):
            if plane_dir.is_dir():
                for fname in ["iscell.npy", "ops.npy", "stat.npy"]:
                    fpath = plane_dir / fname
                    if fpath.exists():
                        paths_to_copy.add(fpath)
                        
    # 3. suite2p_traces with ast_neuropil:False at recording level
    # Some sessions have suite2p_traces registered directly under recordings (not session)
    # with ast_neuropil:False — these also need their plane dff/spks/etc files copied.
    session_entity = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flm_sess
    )
    if session_entity is not None:
        recordings = flz.get_children(
            parent_id=session_entity.id,
            flexilims_session=flm_sess,
            children_datatype="recording",
        )
        if recordings is not None and not recordings.empty:
            for _, rec in recordings.iterrows():
                rec_traces = flz.get_datasets(
                    origin_name=rec["name"],
                    dataset_type="suite2p_traces",
                    filter_datasets={"ast_neuropil": False},
                    flexilims_session=flm_sess,
                    allow_multiple=True,
                )
                if rec_traces is None:
                    continue
                if isinstance(rec_traces, pd.DataFrame):
                    rec_traces_list = [flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess) for _, ds in rec_traces.iterrows()]
                elif isinstance(rec_traces, pd.Series):
                    rec_traces_list = [flz.Dataset.from_flexilims(id=ds.id, flexilims_session=flm_sess) for ds in rec_traces]
                elif isinstance(rec_traces, list):
                    rec_traces_list = rec_traces
                else:
                    rec_traces_list = [rec_traces]
                for traces_ds in rec_traces_list:
                    try:
                        rel = traces_ds.path_full.relative_to(processed_root_flz)
                        src_traces_path = src_processed_root / rel
                    except ValueError:
                        src_traces_path = traces_ds.path_full
                    for plane_dir in sorted(src_traces_path.glob("plane*")):
                        if plane_dir.is_dir():
                            for fname in ["iscell.npy", "ops.npy", "stat.npy", "dff.npy", "dff_ast.npy", "spks.npy"]:
                                fpath = plane_dir / fname
                                if fpath.exists():
                                    paths_to_copy.add(fpath)

    # 4. monitor_frames_df
    monitor_frames_ds = flz.get_datasets(
        origin_name=session_name,
        dataset_type="monitor_frames_df",
        flexilims_session=flm_sess,
        allow_multiple=True,
    )
    if monitor_frames_ds is not None:
        if isinstance(monitor_frames_ds, pd.DataFrame):
            for _, ds in monitor_frames_ds.iterrows():
                if hasattr(ds, "path_full"):
                    try:
                        rel = ds.path_full.relative_to(processed_root_flz)
                        src_ds_path = src_processed_root / rel
                    except ValueError:
                        src_ds_path = ds.path_full
                    if src_ds_path.exists():
                        paths_to_copy.add(src_ds_path)
        else:
            for ds in monitor_frames_ds:
                if hasattr(ds, "path_full"):
                    try:
                        rel = ds.path_full.relative_to(processed_root_flz)
                        src_ds_path = src_processed_root / rel
                    except ValueError:
                        src_ds_path = ds.path_full
                    if src_ds_path.exists():
                        paths_to_copy.add(src_ds_path)

    return paths_to_copy, suite2p_suffix

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
            if hasattr(dataset_obj, "path_full"):
                # When running in --offline mode, dataset_obj.path_full resolves to local BlackPasspo path.
                # Since the folder doesn't exist locally yet, we must check existence on the mounted server volumes first.
                path_exists = False
                if hasattr(dataset_obj, "path") and dataset_obj.path:
                    for s_root in [
                        Path("/Volumes/proj-znamenp-3dvision/raw"),
                        Path("/Volumes/lab-znamenskiyp/home/shared/projects"),
                        Path("/Volumes/lab-znamenskiyp/data/instruments/raw_data/projects"),
                    ]:
                        if (s_root / dataset_obj.path).exists():
                            path_exists = True
                            break
                if not path_exists:
                    path_exists = dataset_obj.path_full.exists()
                    
                if path_exists:
                    paths_to_copy.add(dataset_obj.path_full)
                
    return paths_to_copy

def resolve_src_and_target(p, db_processed_root, db_raw_root, src_processed_root, src_raw_root, dest_processed, dest_raw):
    p = Path(p)
    # Check if p is relative to db_processed_root
    try:
        rel = p.relative_to(db_processed_root)
        return src_processed_root / rel, dest_processed / rel, "processed"
    except ValueError:
        pass
    
    # Check if p is relative to db_raw_root
    try:
        rel = p.relative_to(db_raw_root)
        return src_raw_root / rel, dest_raw / rel, "raw"
    except ValueError:
        pass

    # Check if p is already relative to server roots (just in case)
    try:
        rel = p.relative_to(src_processed_root)
        return p, dest_processed / rel, "processed"
    except ValueError:
        pass

    try:
        rel = p.relative_to(src_raw_root)
        return p, dest_raw / rel, "raw"
    except ValueError:
        pass

    return None, None, None

def main():
    parser = argparse.ArgumentParser(description="Copy raw behavioral and processed session traces for V1 depth map figures to BlackPasspo.")
    parser.add_argument("dest", type=str, default="/Volumes/BlackPasspo/v1_depth_map", nargs="?", help="Destination root directory for figures project.")
    parser.add_argument("--revisions-dest", type=str, default=None, help="Destination root directory for treadmill/revisions project. Defaults to the same as dest.")
    parser.add_argument("--src-project", type=str, default="hey2_3d-vision_foodres_20220101", help="Flexilims project name.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the paths that would be copied without copying.")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode using the local offline database JSON.")
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
    flm_sess = flz.get_flexilims_session(project_id=args.src_project, offline_mode=args.offline)
    
    # Retrieve DB paths (which may be redirected to local BlackPasspo)
    processed_root_flz = flz.get_data_root("processed", flexilims_session=flm_sess)
    raw_root_flz = flz.get_data_root("raw", flexilims_session=flm_sess)
    
    # Real network server paths
    processed_root = Path("/Volumes/lab-znamenskiyp/home/shared/projects")
    raw_root = Path("/Volumes/proj-znamenp-3dvision/raw")

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
    from cottage_analysis.summary_analysis import get_session_list
    print("Fetching list of V1 sessions used in notebooks...")
    notebook_sessions = get_session_list.get_sessions(
        flm_sess,
        exclude_openloop=False,
        exclude_pure_closedloop=False,
        v1_only=True
    )
    print(f"Found {len(notebook_sessions)} sessions used in notebooks.")
    
    from tqdm import tqdm
    print("Discovering figures project session datasets...")
    for sess_name in tqdm(notebook_sessions, desc="Discovering figures", unit="session"):
        try:
            # A. Processed copy
            processed_paths, s2p_suffix = copy_processed_session(flm_sess, sess_name, dest_processed)
            if processed_paths:
                for p in processed_paths:
                    src_p, target, _ = resolve_src_and_target(
                        p, processed_root_flz, raw_root_flz, 
                        processed_root, raw_root, dest_processed, dest_raw
                    )
                    if src_p:
                        all_paths_to_copy.append((src_p, target, "file"))

            # B. Raw metadata & video copy
            raw_paths = copy_raw_session(flm_sess, sess_name, dest_raw, ["SpheresPermTubeReward", "SizeControl"])
            for p in raw_paths:
                src_p, target, _ = resolve_src_and_target(
                    p, processed_root_flz, raw_root_flz, 
                    processed_root, raw_root, dest_processed, dest_raw
                )
                if not src_p:
                    print(f"WARNING: Path {p} is outside raw and processed roots. Skipping.")
                    continue

                if src_p.is_file():
                    # Avoid copying heavy videos unless it's the exact eye tracking calibration session
                    if "video_file" in src_p.name or src_p.suffix in [".mp4", ".avi", ".tif"]:
                        if sess_name != "PZAG3.4f_S20220421": 
                            continue
                    all_paths_to_copy.append((src_p, target, "file"))
                elif src_p.is_dir():
                    all_paths_to_copy.append((src_p, target, ("custom_dir", s2p_suffix)))
        except Exception as e:
            if "Project must match" in str(e):
                continue
            raise

    # Revisions project discovery
    print("Connecting to flexilims for revisions project...")
    try:
        from v1_depth_map.revisions.revision_sessions import sessions as rev_sessions
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
        from v1_depth_map.revisions.revision_sessions import sessions as rev_sessions

    flm_rev = flz.get_flexilims_session(project_id="colasa_3d-vision_revisions", offline_mode=args.offline)
    processed_root_rev_flz = flz.get_data_root("processed", flexilims_session=flm_rev)
    raw_root_rev_flz = flz.get_data_root("raw", flexilims_session=flm_rev)
    
    processed_root_rev = Path("/Volumes/lab-znamenskiyp/home/shared/projects")
    raw_root_rev = Path("/Volumes/lab-znamenskiyp/data/instruments/raw_data/projects")

    print("Discovering revisions project session datasets...")
    for sess_name, protocol in tqdm(rev_sessions.items(), desc="Discovering revisions", unit="session"):
        try:
            # A. Processed copy
            processed_paths, s2p_suffix = copy_processed_session(flm_rev, sess_name, dest_rev_processed)
            if processed_paths:
                for p in processed_paths:
                    src_p, target, _ = resolve_src_and_target(
                        p, processed_root_rev_flz, raw_root_rev_flz, 
                        processed_root_rev, raw_root_rev, dest_rev_processed, dest_rev_raw
                    )
                    if src_p:
                        all_paths_to_copy.append((src_p, target, "file"))

            # B. Raw metadata copy (no videos/tiffs needed for revision notebooks)
            raw_paths = copy_raw_session(flm_rev, sess_name, dest_rev_raw, ["SpheresTubeMotor", "SpheresPermTubeReward"])
            for p in raw_paths:
                src_p, target, _ = resolve_src_and_target(
                    p, processed_root_rev_flz, raw_root_rev_flz, 
                    processed_root_rev, raw_root_rev, dest_rev_processed, dest_rev_raw
                )
                if not src_p:
                    print(f"WARNING: Path {p} is outside raw and processed roots. Skipping.")
                    continue

                if src_p.is_file():
                    if "video_file" in src_p.name or src_p.suffix in [".mp4", ".avi", ".tif"]:
                        continue
                    all_paths_to_copy.append((src_p, target, "file"))
                elif src_p.is_dir():
                    all_paths_to_copy.append((src_p, target, ("custom_dir", s2p_suffix)))
        except Exception as e:
            if "Project must match" in str(e):
                continue
            raise

    print(f"Total paths discovered: {len(all_paths_to_copy)}")


    if args.dry_run:
        print("\n--- Dry Run: Discovery Analysis ---")
        for src, target, mode in sorted(all_paths_to_copy):
            status = " [FOUND]" if src.exists() else " [MISSING]"
            print(f"{status} {src} -> {target} ({mode})")
        return

    # Copy files
    print(f"Syncing files to {dest}...")
    from tqdm import tqdm
    pbar = tqdm(sorted(all_paths_to_copy), desc="Syncing files", unit="item")
    for src, target, mode in pbar:
        pbar.set_description(f"Syncing {src.name}")
        if not src.exists():
            pbar.write(f"WARNING: Source {src} does not exist. Skipping.")
            continue

        if mode == "file":
            target.parent.mkdir(parents=True, exist_ok=True)
            if args.skip_existing and target.exists():
                continue
            robust_copy2(src, target)

        elif mode == "dir":
            if args.skip_existing and target.exists():
                continue
            robust_copytree(src, target)

        elif isinstance(mode, tuple) and mode[0] == "custom_dir":
            s2p_suffix = mode[1]
            target.mkdir(parents=True, exist_ok=True)
            
            # Determine session name from source path (e.g. .../mouse/session_date/recording)
            try:
                mouse_name = src.parent.parent.name
                sess_date = src.parent.name
                sess_name = f"{mouse_name}_{sess_date}"
            except Exception:
                sess_name = ""

            # PZAH6.4b and PZAG3.4f use photodiode_protocol=2, which reads mouse-Z
            # directly from RotaryEncoder.csv instead of FrameLog.csv.
            # All other mice use protocol=5 and only need FrameLog/NewParams/ParamLog.
            PROTOCOL2_MICE = {"PZAH6.4b", "PZAG3.4f"}
            mouse_name_for_check = src.parent.parent.name  # .../mouse/session/recording
            allowed_filenames = {
                "FrameLog.csv", "NewParams.csv", "ParamLog.csv", "harpmessage.npz",
            }
            if mouse_name_for_check in PROTOCOL2_MICE:
                allowed_filenames.add("RotaryEncoder.csv")
            # eye-tracking calibration session also needs camera timestamps
            if sess_name == "PZAG3.4f_S20220421":
                allowed_filenames |= {
                    "left_eye_camera_timestamps.csv", "right_eye_camera_timestamps.csv",
                    "face_camera_timestamps.csv", "butt_camera_timestamps.csv",
                }

            prefix = f"{sess_name}_{src.name}_"
            copied_count = 0
            if src.exists():
                for p_file in src.iterdir():
                    if p_file.is_file():
                        fname = p_file.name
                        if fname.startswith("._"):
                            continue
                        # Strip recording prefix if present (common in older protocol-2 mice)
                        clean_fname = fname
                        if sess_name and fname.startswith(prefix):
                            clean_fname = fname[len(prefix):]

                        is_allowed = (
                            clean_fname in allowed_filenames
                            or clean_fname.startswith("NewParams_")
                            or clean_fname.startswith("ParamLog_")
                        )
                        if is_allowed:
                            target_file = target / fname
                            if args.skip_existing and target_file.exists():
                                continue
                            robust_copy2(p_file, target_file)
                            copied_count += 1
                        
            # Sync suite2p trace subfolders
            # Only copy the subfolder that matches the queried suite2p_suffix!
            if s2p_suffix is not None:
                allowed_subs = [f"suite2p_traces_{s2p_suffix}", f"suite2p_traces_annotated_{s2p_suffix}"]
            else:
                allowed_subs = ["suite2p_traces_0", "suite2p_traces_annotated_0"]

            for sub in allowed_subs:
                src_sub = src / sub
                target_sub = target / sub
                if src_sub.exists():
                    if args.skip_existing and target_sub.exists():
                        continue
                    robust_copytree(src_sub, target_sub, ignore=suite2p_traces_ignore)
            
    print("Data sync successfully completed!")

if __name__ == "__main__":
    main()
