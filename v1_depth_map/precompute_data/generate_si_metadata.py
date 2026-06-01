"""For some analysis we want to read the tiff metadata (to find z for instance)
but we don't want to have to copy that Tb of tiff just for that.
"""

import flexiznam as flz
import numpy as np
from tifffile import TiffFile
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def generate_si_metadata_for_session(flm_sess, session_name):
    """Generate si_metadata.npy for a given session by extracting it from raw TIF files if they exist."""
    print(f"\nProcessing session {session_name}...")

    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flm_sess
    )
    if exp_session is None:
        print(f"  [ERROR] Session {session_name} not found in flexilims.")
        return

    recordings = flz.get_children(
        parent_id=exp_session.id,
        flexilims_session=flm_sess,
        children_datatype="recording",
    )
    if recordings is None or recordings.empty:
        print("  [WARNING] No recordings found for this session.")
        return

    # Only process recordings that are SpheresPermTubeReward or SpheresTubeMotor
    valid_recordings = recordings[
        recordings.name.str.contains("SpheresPermTubeReward|SpheresTubeMotor")
    ]
    if valid_recordings.empty:
        print("  [WARNING] No valid closed-loop recordings found.")
        return

    raw_root = flz.get_data_root("raw", flexilims_session=flm_sess)
    for _, rec in valid_recordings.iterrows():
        print(f"  Recording: {rec['name']}")

        # 1. Fetch scanimage dataset to find tif files
        datasets = flz.get_children(
            parent_id=rec.id,
            flexilims_session=flm_sess,
            children_datatype="dataset",
            filter={"dataset_type": "scanimage"},
        )
        if datasets is None or datasets.empty:
            print("    [WARNING] No ScanImage dataset found.")
            continue

        dataset = datasets.iloc[0]
        if not dataset["tif_files"]:
            print("    [WARNING] No TIF files registered in dataset.")
            continue

        tif_filename = sorted(dataset["tif_files"])[0]
        tif_path = raw_root / rec["path"] / tif_filename

        if not tif_path.exists():
            print(f"    [ERROR] Raw TIF file not found on disk at {tif_path}")
            continue

        # 2. Extract ScanImage metadata
        print("    [SUCCESS] Raw TIF found. Extracting ScanImage metadata...")
        try:
            metadata = TiffFile(str(tif_path)).scanimage_metadata
        except Exception as e:
            print(f"    [ERROR] Failed to extract metadata from TIF: {e}")
            continue

        # 3. Find processed suite2p datasets and save si_metadata.npy
        suite2p_datasets = flz.get_children(
            parent_id=rec.id,
            flexilims_session=flm_sess,
            children_datatype="dataset",
            filter={"dataset_type": "suite2p_traces"},
        )

        if suite2p_datasets is None or suite2p_datasets.empty:
            print("    [WARNING] No suite2p_traces datasets found for this recording.")
            continue

        for _, s2p_ds_row in suite2p_datasets.iterrows():
            s2p_ds = flz.Dataset.from_flexilims(
                id=s2p_ds_row.id, flexilims_session=flm_sess
            )
            s2p_path = s2p_ds.path_full

            if s2p_path.exists():
                meta_file_path = s2p_path / "si_metadata.npy"
                np.save(meta_file_path, metadata)
                print(f"    [SUCCESS] Saved si_metadata.npy to {meta_file_path}")
            else:
                print(f"    [WARNING] Suite2p directory does not exist at {s2p_path}")


def main():
    project = "hey2_3d-vision_foodres_20220101"
    print(f"Connecting to flexilims (project: {project})...")
    flm_sess = flz.get_flexilims_session(project_id=project)

    # We load all sessions in our dataset
    processed_root = flz.get_data_root("processed", flexilims_session=flm_sess)
    pickle_path = (
        processed_root / "v1_manuscript_figures/ver_rev1/fig1/results_all_psth.pickle"
    )
    df = pd.read_pickle(pickle_path)
    session_list = sorted(df["session"].unique())
    print(f"Found {len(session_list)} unique sessions in {pickle_path.name}.")

    for session_name in session_list:
        try:
            generate_si_metadata_for_session(flm_sess, session_name)
        except Exception as e:
            print(f"  [CRITICAL ERROR] Failed processing session {session_name}: {e}")


if __name__ == "__main__":
    main()
