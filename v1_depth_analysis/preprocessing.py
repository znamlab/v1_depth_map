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
from v1_depth_analysis.config import PROJECT, DLC_MODEL

# get a list of all existing sessions
flm_sess = flz.get_flexilims_session(project_id=PROJECT)
sessions = vda.get_sessions(flm_sess=flm_sess)

REDO = False
for sess_name, sess in sessions.iterrows():
    print(f"Processing {sess_name}")
    camera_ds_by_rec = flz.get_datasets(
        sess.id,
        dataset_type="camera",
        flexilims_session=flm_sess,
        return_paths=False,
    )
    camera_ds = []
    for rec_id, camera_ds in camera_ds_by_rec.items():
        camera_ds += camera_ds

    log_df = []
    for cam_ds in camera_ds:
        # get only eye cam
        if not "eye_camera" in cam_ds.dataset_name:
            continue
        log = dict(dataset_name=cam_ds.full_name)

        # run uncropped DLC
        job_id = eye_tracking.run_dlc(
            cam_ds, flm_sess, dlc_model=DLC_MODEL, crop=False, redo=REDO
        )
        log["dlc_uncropped"] = job_id if job_id is not None else "Done"

        # run cropped DLC
        job_id = eye_tracking.run_dlc(
            cam_ds,
            flm_sess,
            dlc_model=DLC_MODEL,
            crop=True,
            redo=REDO,
            job_dependency=job_id,
        )
        log["dlc_cropped"] = job_id if job_id is not None else "Done"
        log_df.append(log)

        # run ellipse fit
        process = eye_tracking.run_fit_ellipse(
            cam_ds,
            flm_sess,
            likelihood_threshold=None,
            job_dependency=job_id,
            use_slurm=True,
        )
        log["ellipse"] = job_id if job_id is not None else "Done"
log_df = pd.DataFrame(log_df)
log_df.to_csv(
    Path(flz.PARAMETERS["data_root"]["processed"]) / PROJECT / "preprocessing_log.csv",
    index=False,
)
