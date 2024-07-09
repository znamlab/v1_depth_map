import flexiznam as flz
from v1_depth_analysis.v1_manuscript_2023 import get_session_list
from v1_depth_analysis.eye_tracking.analysis import get_data
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
all_gaze_data = {}
for session_name in sessions:
    mouse, sess = session_name.split("_")
    if mouse not in ["PZAH6.4b", "PZAG3.4f"]:
        continue
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
        rec = recording[len(session_name) + 1 :]
        gaze_data = get_data(PROJECT, mouse, sess, rec)
        rec_name = f"{session_name}_{recording}"
        gaze_data["rec_name"] = rec_name
        gaze_data["mouse"] = mouse
        gaze_data["session"] = sess
        all_gaze_data[rec_name] = gaze_data
# %%


fig = plt.figure(figsize=(10, 10))
n = len(all_gaze_data)
for i, (k, gaze_data) in enumerate(all_gaze_data.items()):
    if i == n // 3:
        break
    valid = gaze_data[gaze_data["is_stim"] & (~np.isnan(gaze_data["azimuth_filt"]))]
    plt.hist(valid["azimuth"], bins=100, histtype="step", label=k)
plt.legend()

# %%
all_df = pd.concat(all_gaze_data.values())


# %%
gpby = all_df[["rec_name", "azimuth_filt"]].groupby("rec_name")
std = gpby.aggregate(np.nanstd)
mean = gpby.aggregate(np.nanmean)

plt.scatter(mean, std)
