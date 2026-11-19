"""Significance threshold of the closed-loop RS/OF fits, shared by every figure that uses it.

The threshold is the 95th percentile of an empirical null fitted to the held-out R² of each
model (`cottage_analysis.analysis.common_utils.add_rsq_significance`), over the `iscell`
ROIs of the figure 2 sessions. Figure 2 and Fig S5 compute it on the table they load; Fig S10
and `precompute_data/sftf_grating_cells.py` read the same table from disk with
`figure2_rsof_threshold`. All of them go through `add_rsof_significance`, so the value can
only differ if the table on disk differs from the one the figures used.

The table itself (`FIG2_RSOF_TABLE`) is built by `build_figure2_rsof_table` and written
only by `figure2_rsof_integration.ipynb`; every other user reads it with
`load_figure2_rsof_table`.
"""

from pathlib import Path

import pandas as pd
from cottage_analysis.analysis import common_utils

RSOF_MODELS = ("grs", "gof", "gratio", "gadd", "g2d")
RSOF_SIG_PERCENTILE = 95
FIG2_RSOF_TABLE = (
    "fig2/results_all_rsof_closedloop.parquet"  # under the figures read root
)
# Mice of the closed-loop RS/OF analysis (figure 2 and Fig S5)
FIG2_MICE = (
    "PZAH6.4b",
    "PZAG3.4f",
    "PZAH8.2h",
    "PZAH8.2i",
    "PZAH8.2f",
    "PZAH10.2d",
    "PZAH10.2f",
)


def build_figure2_rsof_table(flexilims_session, models=RSOF_MODELS, verbose=True):
    """Concatenate the closed-loop depth and RS/OF fits of every figure 2 session.

    Sessions are the V1 sessions of `FIG2_MICE` with at least 10 trials (closed loop, with
    or without open loop). All ROIs are kept, with their `iscell` flag. Units are
    converted for the cross-validated preferences: RS to cm/s, depth to cm, OF to deg/s.

    Args:
        flexilims_session: flexilims session of the main project.
        models (iterable): RS/OF models whose test R², R² and parameters are kept.
        verbose (bool): passed to `concatenate_all_neurons_df`.

    Returns:
        pd.DataFrame: the table saved as `FIG2_RSOF_TABLE`.
    """
    import flexiznam as flz
    import numpy as np
    from cottage_analysis.summary_analysis import get_session_list, summary_utils

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
    for model in models:
        for col in [
            "rsof_test_rsq_closedloop_",
            "rsof_rsq_closedloop_",
            "rsof_popt_closedloop_",
        ]:
            cols.append(f"{col}{model}")

    mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)
    mouse_list = mouse_list[mouse_list.name.isin(FIG2_MICE)]
    results_all = summary_utils.concatenate_all_neurons_df(
        flexilims_session=flexilims_session,
        session_list=get_session_list.get_sessions(
            flexilims_session,
            exclude_openloop=False,
            exclude_pure_closedloop=False,
            v1_only=True,
            trialnum_min=10,
            mouse_list=mouse_list,
        ),
        filename="neurons_df.pickle",
        cols=cols,
        read_iscell=True,
        verbose=verbose,
    )
    results_all["preferred_RS_closedloop_crossval_g2d"] = (
        results_all["preferred_RS_closedloop_crossval_g2d"] * 100
    )
    results_all["preferred_depth_closedloop_crossval"] = (
        results_all["preferred_depth_closedloop_crossval"] * 100
    )
    results_all["preferred_OF_closedloop_crossval_g2d"] = np.degrees(
        results_all["preferred_OF_closedloop_crossval_g2d"]
    )
    cols_to_keep = [
        col
        for col in results_all.columns
        if not any(x in col for x in ["rf_coef", "dff_stim"])
    ]
    return results_all[cols_to_keep].copy()


def load_figure2_rsof_table(read_root, columns=None):
    """Read `FIG2_RSOF_TABLE` under the figures read root.

    Args:
        read_root (Path): figures read root (`get_figures_roots(...)[0]`).
        columns (list, optional): columns to read. Defaults to all.

    Returns:
        pd.DataFrame: all ROIs, with their `iscell` flag.
    """
    path = Path(read_root) / FIG2_RSOF_TABLE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: run figure2_rsof_integration.ipynb first, it builds "
            "this table"
        )
    return pd.read_parquet(path, columns=columns)


def rsof_rsq_col(model):
    """Held-out R² column of a closed-loop RS/OF model."""
    return f"rsof_test_rsq_closedloop_{model}"


def add_rsof_significance(results_all, models=RSOF_MODELS, verbose=True):
    """Add `<rsq col>_sig` columns in place and return {model: threshold}.

    Args:
        results_all (pd.DataFrame): `iscell` ROIs only, so the null is fitted on cells.
        models (iterable): RS/OF models to threshold.
        verbose (bool): print the threshold and fraction passing per model.

    Returns:
        dict: threshold of each model.
    """
    if "iscell" in results_all.columns and not results_all["iscell"].astype(bool).all():
        raise ValueError("Keep only iscell ROIs before fitting the RS/OF null")
    return common_utils.add_rsq_significance(
        results_all,
        {model: rsof_rsq_col(model) for model in models},
        percentile=RSOF_SIG_PERCENTILE,
        verbose=verbose,
    )


def figure2_rsof_threshold(read_root, model="g2d"):
    """Threshold of one model, computed from the figure 2 table under `read_root`.

    Args:
        read_root (Path): figures read root (`get_figures_roots(...)[0]`).
        model (str): RS/OF model.

    Returns:
        float: threshold, identical to the one figure 2 uses.
    """
    table = load_figure2_rsof_table(read_root, columns=["iscell", rsof_rsq_col(model)])
    table = table[table["iscell"].astype(bool)].copy()
    return float(add_rsof_significance(table, [model], verbose=False)[model])
