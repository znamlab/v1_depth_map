import flexiznam as flz
from cottage_analysis.summary_analysis import depth_responses
from cottage_analysis.summary_analysis import get_session_list
from v1_depth_map.paths import get_figures_roots
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Get PSTH for all sessions #THIS WILL TAKE REALLY LONG
project = "hey2_3d-vision_foodres_20220101"
flexilims_session = flz.get_flexilims_session(project)
mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)
mouse_list = mouse_list[
    mouse_list.name.isin(
        [
            "PZAH6.4b",
            "PZAG3.4f",
            "PZAH8.2h",
            "PZAH8.2i",
            "PZAH8.2f",
            "PZAH10.2d",
            "PZAH10.2f",
        ]
    )
]
session_list = get_session_list.get_sessions(
    flexilims_session,
    exclude_openloop=False,
    exclude_pure_closedloop=False,
    v1_only=True,
    trialnum_min=10,
    mouse_list=mouse_list,
)
results_all = depth_responses.get_psth_crossval_all_sessions(
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
_, SAVE_ROOT = get_figures_roots(flexilims_session, fig_subdir="fig1")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
results_all.to_pickle(SAVE_ROOT / "results_all_psth.pickle")
