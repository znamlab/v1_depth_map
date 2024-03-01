# %%
# select session
import seaborn as sns
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import flexiznam as flz
from v1_depth_analysis.config import PROJECT, MOUSE_LIST
import v1_depth_analysis as vda
from cottage_analysis import eye_tracking
from cottage_analysis.eye_tracking import analysis as analeyesis

# %% [markdown]
# # Get data
#
# ## Find sessions

# %%
flm_sess = flz.get_flexilims_session(project_id=PROJECT)
datasets = []
for mouse in MOUSE_LIST:
    print(f"Getting datasets for {mouse}", flush=True)
    ds_dict = flz.get_datasets_recursively(
        flexilims_session=flm_sess, dataset_type="camera", origin_name=mouse
    )
    for origin, ds in ds_dict.items():
        datasets.extend(ds)
print(f"Found {len(datasets)} datasets. Filtering for eye camera datasets")
eye_cam = [ds for ds in datasets if "eye" in ds.dataset_name.lower()]
print(f"Found {len(eye_cam)} eye camera datasets")
right_eye_cam = [ds for ds in eye_cam if "right" in ds.dataset_name.lower()]
print(f"Found {len(right_eye_cam)} right eye camera datasets")
sphere_right_eye_cam = [ds for ds in right_eye_cam if "SpheresPerm" in ds.full_name]
print(f"Found {len(sphere_right_eye_cam)} right eye camera datasets in Spheres protocols")

# %%
# PURGE. Option to delete all processed data to redo
if False:
    for eye_ds in eye_cam:
        print("Doing", eye_ds.full_name)
        eye_tracking.eye_tracking.clear_tracking_info(eye_ds, flm_sess)
print('done')
# %% [markdown]
# # Preprocess
# %%
DLC_MODEL_DETECT = "headfixed_detect_eye"
DLC_MODEL_TRACKING = "headfixed_track_eye"
CONFLICTS = "overwrite"
USE_SLURM = True

for eye_ds in sphere_right_eye_cam:
    print("Doing", eye_ds.full_name)
    log = eye_tracking.run_all(
        camera_ds_name=eye_ds.full_name,
        origin_id=eye_ds.origin_id,
        flexilims_session=flm_sess,
        dlc_model_detect=DLC_MODEL_DETECT,
        dlc_model_tracking=DLC_MODEL_TRACKING,
        conflicts=CONFLICTS,
        use_slurm=USE_SLURM,
        dependency=None,
        run_detect=True,
        run_tracking=True,
        run_ellipse=True,
        run_reprojection=True,
    )
crash
# %%
# optional: copy all diagnostic plots in a scracth folder, renaming them with the dataset name
base = flz.get_data_root("processed", flexilims_session=flm_sess) / PROJECT
target_dir = Path("/nemo/lab/znamenskiyp/scratch/eye_tracking")
target_dir.mkdir(exist_ok=True)

file_list = target_dir / "list_of_file_names.txt"
import os

os.system(f"find {base} -name 'diagnostic_cropping.png' > {file_list}")

with open(file_list, "r") as f:
    files = f.readlines()
files = [Path(f.strip()) for f in files]
import shutil

target_dir = target_dir / "diagnostic_cropping"
target_dir.mkdir(exist_ok=True)
for f in files:
    shutil.copy(f, target_dir / "_".join(f.relative_to(base).parts))


# %%
d = eye_cam[100]
d.full_name

# %%
d.path_full / d.extra_attributes["video_file"]

# %% [markdown]
# ## Load data

# %%


# %%
tracking_data = dict()
for camera in eye_cam:

    dlc_res, ellipse, dlc_ds = analeyesis.get_data(
        camera,
        flexilims_session=flm_sess,
        likelihood_threshold=0.88,
        rsquare_threshold=0.99,
        error_threshold=3,
    )
    try:
        data, sampling = analeyesis.add_behaviour(
            camera,
            dlc_res,
            ellipse,
            speed_threshold=0.01,
            log_speeds=False,
            verbose=False,
        )
    except FileNotFoundError as err:
        print(err)
        continue
    tracking_data[camera.full_name] = (data, sampling)

# %%
# define depth colors

import matplotlib as mpl
from matplotlib import cm

depths = [np.unique(data.depth) for data, _ in tracking_data.values()]
depth_list = np.unique(np.hstack(depths))
depth_list = depth_list[~np.isnan(depth_list)]
cmap = cm.cool.reversed()
line_colors = []
norm = mpl.colors.Normalize(vmin=np.log(min(depth_list)), vmax=np.log(max(depth_list)))
col_dict = dict()
for depth in depth_list:
    rgba_color = cmap(norm(np.log(depth)), bytes=True)
    rgba_color = tuple(it / 255 for it in rgba_color)
    line_colors.append(rgba_color)
    col_dict[depth] = rgba_color

# %% [markdown]
# # Eye movement
#
# For each session compare the amount of motion per depth

# %%
motion_df = pd.DataFrame(
    columns=np.array(depth_list).astype(int), index=tracking_data.keys()
)
size_df = pd.DataFrame(
    columns=np.array(depth_list).astype(int), index=tracking_data.keys()
)
for sess, (data, sampling) in tracking_data.items():
    data = data[~np.isnan(data.depth) & data.valid]
    avg_by_depth = data.groupby("depth").aggregate(np.mean)
    motion_df.loc[sess, :] = avg_by_depth.pupil_motion
    size_df.loc[sess, :] = avg_by_depth.major_radius * avg_by_depth.minor_radius * np.pi

# %%
sns.displot(size_df.values.flatten())
size_df[np.sum(size_df > 3000, axis=1) > 0]


# %%
sns.displot(motion_df.values.flatten())
motion_df[np.sum(motion_df > 20, axis=1) > 0]


# %%
sns.violinplot(motion_df, palette=line_colors)


# %%
sns.violinplot(size_df, palette=line_colors)


# %%
data.columns


# %%
mvt_angle = np.arctan2(data.delta_position_x, data.delta_position_y)
sns.displot(np.rad2deg(mvt_angle))


# %%
np.hstack([[0], depth_list]) + 1


# %%
import windrose
from windrose import WindroseAxes

ax = WindroseAxes.from_ax()
ax.bar(
    np.rad2deg(mvt_angle) + 90,
    data.depth,
    normed=True,
    opening=1,
    edgecolor="none",
    colors=line_colors,
    bins=depth_list - 1,
)
ax.set_legend()

# %%
for sess, (data, sampling) in tracking_data.items():
    fig, ax = plt.subplots(1, 1)
    data = data[data.valid]
    count, bx, by = np.histogram2d(data.pupil_x, data.pupil_y, bins=(70, 70))
    h, bx, by = np.histogram2d(
        data.pupil_x, data.pupil_y, weights=np.rad2deg(elli.angle), bins=(bx, by)
    )
    h[count < 1] = np.nan
    img = ax.imshow(
        (h / count).T, extent=(bx[0], bx[-1], by[0], by[-1]), vmin=55, vmax=65
    )
    cb = fig.colorbar(mappable=img, ax=ax)
    cb.set_label("Ellipse angle (degrees)")
    ax.set_xlabel("Ellipse pupil X (pixels)")
    ax.set_ylabel("Ellipse pupil Y (pixels)")

# %%
import os
import flexiznam as flz
from pathlib import Path
from v1_depth_analysis import PROJECT
import cv2

processed_path = Path(flz.PARAMETERS["data_root"]["processed"])
calibration_folder = processed_path / PROJECT / "Calibrations"

data = dict()
for camera in ["RightEyeCam", "LeftEyeCam"]:
    data[camera.lower()] = dict()
    folder = calibration_folder / camera
    folder = list(folder.glob("*xtrinsics_flat"))[0]  # case is inconsistent
    folder = folder / "20220818" / "aruco5_5mm"
    assert folder.exists()
    for trial in folder.glob("trial*"):
        fname = str(trial / "camera_extrinsics_flat.yml")
        s = cv2.FileStorage()
        s.open(fname, cv2.FileStorage_READ)
        rvec = s.getNode("rvec").mat()
        tvec = s.getNode("tvec").mat()
        data[camera.lower()][trial.name] = dict(rvec=rvec, tvec=tvec)

# %%
import numpy as np

np.hstack([v["tvec"] for v in data["righteyecam"].values()])

# %%
os.listdir(calibration_folder / "LeftEyeCam")
