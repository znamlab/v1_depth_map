import flexiznam as flz
from cottage_analysis.summary_analysis import get_session_list, rs_stats
from v1_depth_map.paths import get_figures_roots
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

project = "hey2_3d-vision_foodres_20220101"
# must match the filter the figure notebooks pass, see notes/01_preprocessing.md
FILTER_DATASETS = {"anatomical_only": 3, "ast_neuropil": False}
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
    mouse_list=mouse_list,
)
results_all = rs_stats.get_rs_stats_all_sessions(
    flexilims_session,
    session_list,
    nbins=60,
    rs_thr_min=None,
    rs_thr_max=None,
    still_only=False,
    still_time=1,
    corridor_length=6,
    blank_length=3,
    overwrite=True,
    filter_datasets=FILTER_DATASETS,
)

_, SAVE_ROOT = get_figures_roots(flexilims_session, fig_subdir="supp")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)
results_all.to_pickle(SAVE_ROOT / "results_all_rs_supp.pickle")
