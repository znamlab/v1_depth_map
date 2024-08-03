# %%
import flexiznam as flz
import wayla
from v1_depth_analysis.v1_manuscript_2023 import get_session_list

# %%
PROJECT = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(project_id=PROJECT)
sessions = get_session_list.get_sessions(
    flexilims_session=flexilims_session,
    exclude_sessions=(),
    exclude_openloop=True,
    exclude_pure_closedloop=False,
    v1_only=True,
)
print(f"Found {len(sessions)} sessions")
# %%
recording_type = ("two_photon",)
protocol_base = "SpheresPermTubeReward"
for session_name in sessions:
    print(f"Processing {session_name}")
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=exp_session["id"],
        query_key="recording_type",
        query_value=recording_type,
        flexilims_session=flexilims_session,
    )

    recordings = recordings[recordings.name.str.contains(protocol_base)]
    if "exclude_reason" in recordings.columns:
        recordings = recordings[recordings["exclude_reason"].isna()]

    for recording, series in recordings.iterrows():
        print(f"    Processing {recording}")
        wayla.eye_tracking.run_all(
            flexilims_session=flexilims_session,
            dlc_model_detect="headfixed_detect_eye",
            dlc_model_tracking="headfixed_track_eye",
            camera_ds_name=f"{recording}_right_eye_camera",
            origin_id=None,
            conflicts="skip",
            use_slurm=True,
            dependency=None,
            run_detect=False,
            run_tracking=False,
            run_ellipse=False,
            run_reprojection=True,
            repro_kwargs=None,
        )

    # %%
