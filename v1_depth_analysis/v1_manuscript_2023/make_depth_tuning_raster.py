import flexiznam as flz
from v1_depth_analysis.v1_manuscript_2023 import depth_selectivity, get_session_list
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get PSTH for all sessions #THIS WILL TAKE REALLY LONG
project = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(project)

session_list = get_session_list.get_sessions(
    flexilims_session,
    closedloop_only=False,
    openloop_only=False,
)
results_all = depth_selectivity.get_psth_crossval_all_sessions(
    flexilims_session,
    session_list,
    nbins=60,
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
    blank_length=3,
    overwrite=True,
)
VERSION = 6
SAVE_ROOT = (
    Path(
        f"/camp/lab/znamenskiyp/home/shared/presentations/v1_manuscript_2023/ver{VERSION}"
    )
    / "fig1"
)
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
results_all.to_pickle(SAVE_ROOT / "results_all.pickle")