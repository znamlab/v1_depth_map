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
import v1_depth_analysis as vda
from v1_depth_analysis.config import PROJECT, EYE_TRACKING_SESSIONS

REDO = False
DLC_MODEL_DETECT = "headfixed_detect_eye"
DLC_MODEL_TRACKING = "headfixed_track_eye"
CONFLICTS = "overwrite"
USE_SLURM = True

flm_sess = flz.get_flexilims_session(project_id=PROJECT)

log_df = []
for mouse, sess, calib_sess, orientation in EYE_TRACKING_SESSIONS:
    if mouse in ["PZAH6.4b"]:
        continue
    if mouse != "PZAG3.4f":
        continue
    sess_name = f"{mouse}_{sess}"
    sess = flz.get_entity(
        name=sess_name, datatype="session", flexilims_session=flm_sess
    )
    print(f"Processing {sess_name}")

    # get sphere recordings
    sphere_recordings = vda.get_recordings(sessions=pd.DataFrame({sess_name: sess}).T)
    rec = sphere_recordings[0]

    camera_ds = flz.get_datasets(
        dataset_type="camera", origin_id=rec.id, flexilims_session=flm_sess
    )
    camera_ds = [ds for ds in camera_ds if "right" in ds.dataset_name.lower()]
    assert len(camera_ds) == 1, "Multiple right eye cameras found"
    camera_ds = camera_ds[0]

    log = eye_tracking.run_all(
        camera_ds_name=camera_ds.full_name,
        origin_id=camera_ds.origin_id,
        flexilims_session=flm_sess,
        dlc_model_detect=DLC_MODEL_DETECT,
        dlc_model_tracking=DLC_MODEL_TRACKING,
        conflicts=CONFLICTS,
        use_slurm=USE_SLURM,
        dependency=None,
        run_detect=False,
        run_tracking=False,
        run_ellipse=False,
        run_reprojection=True,
        repro_kwargs=dict(likelihood_threshold=0.8),
    )
    log_df.append(log)
