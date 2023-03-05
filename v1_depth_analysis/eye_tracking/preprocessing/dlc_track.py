"""
Preprocessing of eye tracking

This selects the sessions and runs DLC jobs to track eye
"""
import os
from pathlib import Path
import flexiznam as flm
import numpy as np
import yaml
import pandas as pd
from flexiznam.schema import Dataset
from cottage_analysis import eye_tracking
import v1_depth_analysis as vda
from v1_depth_analysis.config import PROJECT, DLC_MODEL


"""
NOTES:

Normal usage is to call this function twice:

- initial tracking:
    REDO = False
    CREATE_CROP_FILE = False
    CROP = False
- refined tracking:
    REDO = True
    CREATE_CROP_FILE = True
    CROP = True
"""
REDO = True  # erase previous tracking and redo
CREATE_CROP_FILE = False  # create crop file. Tracking must have been done once
CROP = True  # track using crop file info


raw_path = Path(flm.PARAMETERS["data_root"]["raw"])
processed_path = Path(flm.PARAMETERS["data_root"]["processed"])
flm_sess = flm.get_flexilims_session(project_id=PROJECT)

recordings = vda.get_recordings(protocol="SpheresPermTubeReward", flm_sess=flm_sess)
datasets = vda.get_datasets(
    recordings, dataset_type="camera", dataset_name_contains="_eye", flm_sess=flm_sess
)

for dataset in datasets:
    target_folder = Path(processed_path, PROJECT, *dataset.genealogy)
    rec = flm.get_entity(id=dataset.origin_id, flexilims_session=flm_sess)
    dlc_datasets = vda.get_datasets(rec, dataset_type="dlc_tracking", flm_sess=flm_sess)
    ds = [
        ds
        for ds in dlc_datasets
        if ds.dataset_name.startswith(f"dlc_tracking_{dataset.dataset_name}")
    ]
    if len(ds) > 1:
        raise IOError("No clue why I hav emore than one file here")
    elif len(ds):
        ds = ds[0]
        if CREATE_CROP_FILE:
            metadata_path = (
                dataset.path_full / dataset.extra_attributes["metadata_file"]
            )
            video_path = dataset.path_full / dataset.extra_attributes["video_file"]
            crop_file = target_folder / f"{video_path.stem}_crop_tracking.yml"
            if crop_file.exists():
                print("Crop file already exists. Delete manually to redo")
            else:
                with open(metadata_path, "r") as fhandle:
                    metadata = yaml.safe_load(fhandle)
                dlc_files = list(target_folder.glob(f"{video_path.stem}*.h5"))
                if not len(dlc_files):
                    raise IOError("Cannot create cropping file without DLC info")
                if len(dlc_files) > 1:
                    raise IOError("Multiple tracking")
                print("Creating crop file")
                dlc_res = pd.read_hdf(dlc_files[0])
                # Find DLC crop area
                borders = np.zeros((4, 2))
                for iw, w in enumerate(
                    (
                        "left_eye_corner",
                        "right_eye_corner",
                        "top_eye_lid",
                        "bottom_eye_lid",
                    )
                ):
                    vals = dlc_res.xs(w, level=1, axis=1)
                    vals.columns = vals.columns.droplevel("scorer")
                    v = np.nanmedian(vals[["x", "y"]].values, axis=0)
                    borders[iw, :] = v

                borders = np.vstack(
                    [np.nanmin(borders, axis=0), np.nanmax(borders, axis=0)]
                )
                borders += ((np.diff(borders, axis=0) * 0.2).T @ np.array([[-1, 1]])).T
                for i, w in enumerate(["Width", "Height"]):
                    borders[:, i] = np.clip(borders[:, i], 0, metadata[w])
                borders = borders.astype(int)
                crop_info = dict(
                    xmin=int(borders[0, 0]),
                    xmax=int(borders[1, 0]),
                    ymin=int(borders[0, 1]),
                    ymax=int(borders[1, 1]),
                    dlc_source=str(dlc_files[0]),
                )
                with open(crop_file, "w") as fhandle:
                    yaml.dump(crop_info, fhandle)
                print("Crop file created")
        if not REDO:
            print("  Already done. Skip")
            continue
        else:
            print("  Erasing previous tracking to redo")
            # delete labeled and filtered version too
            filenames = []
            for suffix in ["", "_filtered"]:
                p = ds.path_full
                basename = p.with_name(p.stem + suffix)
                for ext in [".h5", ".csv"]:
                    filenames.append(basename.with_suffix(ext))
                filenames.append(basename.with_name(basename.stem + "_labeled.mp4"))
            filenames.append(p.with_name(p.stem + "_meta.pickle"))
            for fname in filenames:
                if fname.exists():
                    print(f"        deleting {fname}")
                    os.remove(fname)

    if CROP:
        video_path = dataset.path_full / dataset.extra_attributes["video_file"]
        crop_file = target_folder / f"{video_path.stem}_crop_tracking.yml"
        with open(crop_file, 'r') as fhandle:
            crop_info = yaml.safe_load(fhandle)

    process = eye_tracking.dlc_track(
        video_path=dataset.path_full / dataset.extra_attributes["video_file"],
        model_name=DLC_MODEL,
        target_folder=target_folder,
        origin_id=dataset.origin_id,
        project=PROJECT,
        filter=False,
        label=False,
        crop=CROP,
    )
