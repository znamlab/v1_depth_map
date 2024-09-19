"""
Main script ensuring preprocessing is done before analysis

This will run on all sessions in the database and will check that the following
processing steps have been done:

- run DLC tracking twice (once with cropping, once without)
- fit ellipses to DLC output
- reproject pupil from ellipse fit


"""

from pathlib import Path
import flexiznam as flz
import pandas as pd
from cottage_analysis import eye_tracking
import v1_depth_map as vda
from v1_depth_map.eye_tracking.config import PROJECT, DLC_MODEL

# get a list of all existing sessions
flm_sess = flz.get_flexilims_session(project_id=PROJECT)
sessions = vda.get_sessions(flm_sess=flm_sess)

REDO = False
log_df = []
for sess_name, sess in sessions.iterrows():
    print(f"Processing {sess_name}")
    camera_ds_by_rec = flz.get_datasets(
        session_id=sess.id,
        dataset_type="camera",
        flexilims_session=flm_sess,
        return_paths=False,
    )
    # flatten dict
    camera_ds = []
    for rec_id, camera_ds in camera_ds_by_rec.items():
        camera_ds += camera_ds

    log_df = []
    for cam_ds in camera_ds:
        # get only eye cam
        if not "eye_camera" in cam_ds.dataset_name:
            continue
        log = eye_tracking.run_all(
            camera_ds=cam_ds,
            flexilims_session=flm_sess,
            dlc_model=DLC_MODEL,
            redo=REDO,
            use_slurm=True,
        )
        log_df.append(log)

log_df = pd.concat(log_df, ignore_index=True)
log_df.to_csv(
    Path(flz.PARAMETERS["data_root"]["processed"]) / PROJECT / "preprocessing_log.csv",
    index=False,
)
