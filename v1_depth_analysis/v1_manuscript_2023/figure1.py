import functools

print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pickle
from tqdm import tqdm
import scipy

import flexiznam as flz
from cottage_analysis.analysis import spheres, common_utils
from cottage_analysis.pipelines import pipeline_utils
from v1_depth_analysis.v1_manuscript_2023 import depth_selectivity, get_session_list
from v1_depth_analysis.v1_manuscript_2023 import common_utils as plt_common_utils

VERSION = 1
SAVE_ROOT = (
    "/camp/lab/znamenskiyp/home/shared/presentations/v1_manuscript_2023/ver"
    + str(VERSION)
    + "/fig1/"
)
os.makedirs(SAVE_ROOT, exist_ok=True)

# LOAD DATA
# Load an example session
project = "hey2_3d-vision_foodres_20220101"
session_name = "PZAH8.2h_S20230116"
flexilims_session = flz.get_flexilims_session(project)

vs_df_example, trials_df_example = spheres.sync_all_recordings(
    session_name=session_name,
    flexilims_session=flexilims_session,
    project=project,
    filter_datasets={"anatomical_only": 3},
    recording_type="two_photon",
    protocol_base="SpheresPermTubeReward",
    photodiode_protocol=5,
    return_volumes=True,
)

neurons_ds_example = pipeline_utils.create_neurons_ds(
    session_name=session_name,
    flexilims_session=flexilims_session,
    project=None,
    conflicts="skip",
)
neurons_df_example = pd.read_pickle(neurons_ds_example.path_full)

# Get PSTH for all sessions
project = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(project)
mouse_list = [
    # "PZAH6.4b",
    # "PZAG3.4f",
    "PZAH8.2h",
    "PZAH8.2i",
    "PZAH8.2f",
    "PZAH10.2d",
    "PZAH10.2f",
]
session_list = get_session_list.get_all_sessions(
    project=project, mouse_list=mouse_list, closedloop_only=False, openloop_only=False
)
results_all = depth_selectivity.get_psth_crossval_all_sessions(
    flexilims_session,
    session_list,
    nbins=10,
    closed_loop=1,
    use_cols=[
        "roi",
        "is_depth_neuron",
        "depth_neuron_anova_p",
        "best_depth",
        "preferred_depth_closedloop",
        "depth_tuning_popt_closedloop",
        "depth_tuning_trials_closedloop",
        "depth_tuning_trials_closedloop_crossval",
        "preferred_depth_closedloop_crossval",
        "depth_tuning_test_rsq_closedloop",
        "depth_tuning_test_spearmanr_rval_closedloop",
        "depth_tuning_test_spearmanr_pval_closedloop",
    ],
)

# Get visually responsive neurons
neurons_df_all = depth_selectivity.get_visually_responsive_all_sessions(
    flexilims_session=flexilims_session,
    session_list=get_session_list.get_all_sessions(
        project=project,
        mouse_list=[
            "PZAH6.4b",
            #   "PZAG3.4f",
            "PZAH8.2h",
            "PZAH8.2i",
            "PZAH8.2f",
            "PZAH10.2d",
            "PZAH10.2f",
        ],
        closedloop_only=False,
        openloop_only=False,
    ),
    is_closed_loop=1,
    use_cols=[
        "roi",
        "is_depth_neuron",
        "visually_responsive",
        "depth_neuron_anova_p",
        "best_depth",
        "preferred_depth_closedloop",
        "depth_tuning_popt_closedloop",
        "depth_tuning_trials_closedloop_crossval",
        "preferred_depth_closedloop_crossval",
        "depth_tuning_test_rsq_closedloop",
        "depth_tuning_test_spearmanr_rval_closedloop",
        "depth_tuning_test_spearmanr_pval_closedloop",
    ],
    before_onset=0.5,
    frame_rate=15,
)


fig = plt.figure()

# Fig.1A: Motion parallax schema & equation, VR schema
# Fig.1B: Trial structure schema, Spheres with different sizes schema

# Fig.1C: Raster plot of an example neuron
EXAMPLE_ROI = 250
depth_selectivity.plot_raster_all_depths(
    fig=fig,
    neurons_df=neurons_df_example,
    trials_df=trials_df_example,
    roi=EXAMPLE_ROI,
    is_closed_loop=True,
    max_distance=6,
    nbins=60,
    frame_rate=15,
    vmax=3,
    plot=True,
    plot_x=0,
    plot_y=1,
    plot_width=0.8,
    plot_height=0.2,
    cbar_width=0.01,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
)

# Fig.1D: Example PSTH of a single neuron
depth_selectivity.plot_PSTH(
    fig=fig,
    trials_df=trials_df_example,
    roi=EXAMPLE_ROI,
    is_closed_loop=True,
    max_distance=6,
    nbins=20,
    frame_rate=15,
    plot_x=0,
    plot_y=0.67,
    plot_width=0.2,
    plot_height=0.2,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    linewidth=2,
    legend_on=True,
)

# Fig.1E: Example depth tuning curve & PSTH of a single neuron
depth_selectivity.plot_depth_tuning_curve(
    fig=fig,
    neurons_df=neurons_df_example,
    trials_df=trials_df_example,
    roi=EXAMPLE_ROI,
    rs_thr=None,
    plot_fit=False,
    linewidth=2,
    linecolor="k",
    fit_linecolor="r",
    closed_loop=1,
    plot_x=0.35,
    plot_y=0.67,
    plot_width=0.2,
    plot_height=0.2,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
)


# Fig.1F: Example depth tuning curve & PSTH of 3 other neurons
EXAMPLE_ROIS = [174, 249, 54, 344, 86, 88]
plot_cols = 3
for i, roi in enumerate(EXAMPLE_ROIS):
    depth_selectivity.plot_depth_tuning_curve(
        fig=fig,
        neurons_df=neurons_df_example,
        trials_df=trials_df_example,
        roi=roi,
        rs_thr=None,
        plot_fit=False,
        linewidth=2,
        linecolor="k",
        fit_linecolor="r",
        closed_loop=1,
        plot_x=0.62 + 0.13 * (i % plot_cols),
        plot_y=0.67 + 0.13 - 0.13 * (i // plot_cols),
        plot_width=0.08,
        plot_height=0.08,
        fontsize_dict={"title": 15, "label": 5, "tick": 5},
    )
    if i != (len(EXAMPLE_ROIS) - plot_cols // 2 - 1):
        plt.xlabel("")
    if (i % plot_cols) != 0:
        plt.ylabel("")


# Fig.1G: Histogram of proportion of depth-tuned neurons;
# preferred depths of all depth-tuned neurons;
# Raster plot of preferred depths of all neurons (sorted by preferred depth)
depth_selectivity.plot_depth_neuron_perc_hist(
    fig=fig,
    results_df=neurons_df_all,
    numerator_filter=(
        (neurons_df_all["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
        & (neurons_df_all["visually_responsive"] == 1)
    ),
    denominator_filter=(neurons_df_all["visually_responsive"] == 1),
    bins=30,
    plot_x=0,
    plot_y=0.2,
    plot_width=0.25,
    plot_height=0.3,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
)

depth_selectivity.plot_preferred_depth_hist(
    fig=fig,
    results_df=neurons_df_all[
        neurons_df_all["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05
    ],
    use_col="preferred_depth_closedloop_crossval",
    nbins=50,
    plot_x=0.4,
    plot_y=0.2,
    plot_width=0.25,
    plot_height=0.3,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
)

depth_selectivity.plot_psth_raster(
    fig=fig,
    results_df=results_all,
    depth_list=np.geomspace(5, 640, num=8).astype(int),
    use_cols=["preferred_depth_crossval", "psth_crossval", "preferred_depth_rsq"],
    depth_rsq_thr=0.04,
    plot_x=0.8,
    plot_y=0.2,
    plot_width=0.25,
    plot_height=0.3,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},
)

fig.savefig(SAVE_ROOT + "fig1.pdf", bbox_inches="tight", transparent=True)
