import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # for pdfs
matplotlib.rcParams['svg.fonttype'] = 'none' # for svgs
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy import stats
from pathlib import Path

import flexiznam as flz
from cottage_analysis.analysis import spheres, common_utils
from cottage_analysis.pipelines import pipeline_utils
from v1_depth_analysis.v1_manuscript_2023 import depth_selectivity, closed_loop_rsof, get_session_list, depth_decoder
from v1_depth_analysis.v1_manuscript_2023 import common_utils as v1_common_utils
from tqdm import tqdm

READ_VERSION = 9
READ_ROOT = Path(
    f"/camp/lab/znamenskiyp/home/shared/presentations/v1_manuscript_2023/ver{READ_VERSION}"
)

VERSION = 9
SAVE_ROOT = Path(
    f"/camp/lab/znamenskiyp/home/shared/presentations/v1_manuscript_2023/ver{VERSION}"
)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
(SAVE_ROOT/"fig2").mkdir(parents=True, exist_ok=True)

project = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(project)


# Concatenate all sessions for closedloop rsof
models = ["gof", "grs", "gadd", "g2d", "gratio"]
cols = [
    "roi",
    "best_depth",
    "preferred_depth_closedloop",
    "preferred_depth_closedloop_crossval",
    "depth_tuning_popt_closedloop",
    "depth_tuning_test_rsq_closedloop",
    "depth_tuning_test_spearmanr_pval_closedloop",
    "depth_tuning_test_spearmanr_rval_closedloop",
    "preferred_RS_closedloop_g2d",
    "preferred_RS_closedloop_crossval_g2d",
    "preferred_OF_closedloop_g2d",
    "preferred_OF_closedloop_crossval_g2d",
]
cols_to_add = [
    "rsof_test_rsq_closedloop_",
    "rsof_rsq_closedloop_",
    "rsof_popt_closedloop_",
]
for model in models:
    for col in cols_to_add:
        cols.append(f"{col}{model}")

mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)
mouse_list = mouse_list[mouse_list.name.isin(["PZAH6.4b",
                "PZAG3.4f",
                "PZAH8.2h",
                "PZAH8.2i",
                "PZAH8.2f",
                "PZAH10.2d",
                "PZAH10.2f"])]
results_all = pd.read_pickle(READ_ROOT / "fig2" /  "results_all_rsof_closedloop.pickle")


# comparison between different models
results_all["preferred_depth_amplitude"] = results_all[
    "depth_tuning_popt_closedloop"
].apply(lambda x: np.exp(x[0]) + x[-1])
neurons_df_sig = results_all[
    (results_all["iscell"] == 1)
    & (results_all["depth_tuning_test_spearmanr_rval_closedloop"] > 0.1)
    & (results_all["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05)
    & (results_all["preferred_depth_amplitude"] > 0.5)
] 
neurons_df_sig["mouse"] = neurons_df_sig["session"].str.split("_").str[0]
n_boot = 20000
props_all = np.zeros((n_boot, len(models)))
# Find the best model for each neuron
model_cols = [f"rsof_test_rsq_closedloop_{model}" for model in models]
neurons_df_sig["best_model"] = neurons_df_sig[model_cols].idxmax(axis=1)

for iboot in tqdm(range(n_boot)):
    sample = common_utils.bootstrap_sample(
        neurons_df_sig, ["mouse"]
    )

    # Calculate percentage of neurons that have the best model
    neuron_sum = (
        neurons_df_sig.loc[sample].groupby("session")[["roi"]].agg(["count"]).values.flatten()
    )
    props = []
    # calculate the proportion of neurons that have the best model for each session
    for i, model in enumerate(model_cols):
        prop = (
            neurons_df_sig.loc[sample].groupby("session")
            .apply(lambda x: x[x["best_model"] == model][["roi"]].agg(["count"]))
            .values.flatten()
        ) / neuron_sum
        props.append(prop)
    props_all[iboot,:] = np.median(np.array(props),axis=1)
np.save(SAVE_ROOT / "fig2" / "model_comparison_bootstraps.npy", props_all)

for i,imodel in enumerate([0,1,2,4]):
    # plt.subplot(2,2,i+1)
    # plt.hist(props_all[:,3] - props_all[:,imodel], bins=20)
    print(np.mean((props_all[:,3] - props_all[:,imodel])<0)*2)