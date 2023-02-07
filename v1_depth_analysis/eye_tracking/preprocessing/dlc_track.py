"""
Preprocessing of eye tracking

This selects the sessions and runs DLC jobs to track eye
"""

from pathlib import Path
import flexiznam as flm
from flexiznam.schema import Dataset
from cottage_analysis import eye_tracking
from v1_depth_analysis.config import SESSIONS, PROJECT

REDO = True

raw_path = Path(flm.PARAMETERS["data_root"]["raw"])
processed_path = Path(flm.PARAMETERS["data_root"]["processed"])
flm_sess = flm.get_flexilims_session(project_id=PROJECT)

for mouse, sessions in SESSIONS.items():
    mouse_folder = raw_path / PROJECT / mouse
    assert mouse_folder.is_dir()
    for session in sessions:
        session_folder = mouse_folder / f"S20{session}"
        assert session_folder.is_dir()

        # get recordings
        sess = flm.get_entity(name=f"{mouse}_S20{session}", flexilims_session=flm_sess)
        recs = flm.get_children(
            sess.id, children_datatype="recording", flexilims_session=flm_sess
        )
        for rec_name, rec in recs.iterrows():
            if rec["protocol"] == "SpheresPermTubeReward":
                print(f"Doing {rec_name}")
                datasets = flm.get_children(
                    rec.id, children_datatype="dataset", flexilims_session=flm_sess
                )
                datasets = [
                    Dataset.from_flexilims(data_series=ds, flexilims_session=flm_sess)
                    for _, ds in datasets.iterrows()
                ]
                # keep only eye cam
                datasets = [ds for ds in datasets if ds.dataset_type == "camera"]
                datasets = [ds for ds in datasets if "_eye_" in ds.dataset_name]
                for ds in datasets:
                    target_folder = Path(processed_path, PROJECT, *ds.genealogy)
                    if len(list(target_folder.glob("*.h5"))) and not REDO:
                        print("  Already done. Skip")
                        continue
                    process = eye_tracking.dlc_track(
                        video_path=ds.path_full / ds.extra_attributes["video_file"],
                        model_name=DLC_MODEL,
                        target_folder=target_folder,
                        filter=False,
                        label=True,
                    )
