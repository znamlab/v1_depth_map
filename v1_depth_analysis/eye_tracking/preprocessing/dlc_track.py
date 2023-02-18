"""
Preprocessing of eye tracking

This selects the sessions and runs DLC jobs to track eye
"""
import os
from pathlib import Path
import flexiznam as flm
from flexiznam.schema import Dataset
from cottage_analysis import eye_tracking
import v1_depth_analysis as vda
from v1_depth_analysis.config import PROJECT, DLC_MODEL

REDO = True

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

    for ds in dlc_datasets:
        if ds.dataset_name.startswith(f"dlc_tracking_{dataset.dataset_name}"):
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
                for fname in filenames:
                    if fname.exists():
                        print(f"        deleting {fname}")
                        os.remove(fname)

    process = eye_tracking.dlc_track(
        video_path=dataset.path_full / dataset.extra_attributes["video_file"],
        model_name=DLC_MODEL,
        target_folder=target_folder,
        origin_id=dataset.origin_id,
        project=PROJECT,
        filter=False,
        label=False,
    )
