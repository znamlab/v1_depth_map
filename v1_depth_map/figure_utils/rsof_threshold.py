"""Significance threshold of the closed-loop RS/OF fits, shared by every figure that uses it.

The threshold is the 95th percentile of an empirical null fitted to the held-out R² of each
model (`cottage_analysis.analysis.common_utils.add_rsq_significance`), over the `iscell`
ROIs of the figure 2 sessions. Figure 2 and Fig S5 compute it on the table they load; Fig S10
and `precompute_data/sftf_grating_cells.py` read the same table from disk with
`figure2_rsof_threshold`. All of them go through `add_rsof_significance`, so the value can
only differ if the table on disk differs from the one the figures used.
"""

from pathlib import Path

import pandas as pd
from cottage_analysis.analysis import common_utils

RSOF_MODELS = ("grs", "gof", "gratio", "gadd", "g2d")
RSOF_SIG_PERCENTILE = 95
FIG2_RSOF_TABLE = (
    "fig2/results_all_rsof_closedloop.parquet"  # under the figures read root
)


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
    table = pd.read_parquet(
        Path(read_root) / FIG2_RSOF_TABLE, columns=["iscell", rsof_rsq_col(model)]
    )
    table = table[table["iscell"].astype(bool)].copy()
    return float(add_rsof_significance(table, [model], verbose=False)[model])
