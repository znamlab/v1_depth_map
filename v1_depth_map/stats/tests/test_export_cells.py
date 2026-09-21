"""Execute every notebook's stats export cell against stub data.

No 2P caches, no external drive, no kernel: each export cell is sliced from its
"# Export statistics" marker and exec'd with stand-in variables. This is what catches
a broken export before a multi-hour notebook run does - in particular
`f"{r:.3f}"` on the ndarray `hierarchical_bootstrap_stats` returns, which raises
TypeError and leaves the previous (possibly hand-entered) YAML silently in place.
"""

import ast
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
MARKER = "# Export statistics for manuscript tracking"
UNESCAPED_DOLLAR = re.compile(r"(?<!\\)\$")

SESSIONS = ["PZAH6.4b_S1", "PZAG3.4f_S2", "PZAH8.2h_S3", "PZAH8.2i_S4"]


def _neurons(n, extra=None):
    """A stand-in neurons_df with the columns the export cells touch."""
    df = pd.DataFrame(
        {
            "session": [SESSIONS[i % len(SESSIONS)] for i in range(n)],
            "is_depth_neuron": np.arange(n) % 3 == 0,
        }
    )
    df["mouse"] = df["session"].str.split("_").str[0]
    for col, value in (extra or {}).items():
        df[col] = value(n) if callable(value) else value
    return df


def _describe(values):
    return pd.Series(values).describe()


def _stubs(name):
    rng = np.random.default_rng(0)
    if name == "figure1_depth_selectivity":
        # 19 sessions across 4 mice, matching the multi-day cohort the export counts
        sessions_md = [f"MOUSE{m}_S{s}" for m in range(4) for s in range(5)][:19]
        neurons_df_multiday = pd.DataFrame({"session": sessions_md})
        neurons_df_multiday["mouse"] = (
            neurons_df_multiday["session"].str.split("_").str[0]
        )
        # one row per tracked neuron, one column per day; NaN where it was not tracked,
        # so the export's dropna() pairing is actually exercised
        day_values = rng.normal(size=(300, 5))
        day_values[rng.random((300, 5)) < 0.3] = np.nan
        return dict(
            results_all=_neurons(400),
            neurons_df_all=_neurons(59937),
            neurons_df_multiday=neurons_df_multiday,
            df_pivot_multiday=pd.DataFrame(day_values, columns=[1, 2, 3, 4, 5]),
            x_vals=list(range(345)),
        )
    if name == "figure_openloop":
        return dict(
            neurons_df_sig_openloop=_neurons(1229),
            neurons_df_sig_openloop_depth_selective=_neurons(9429),
            decoder_results=_neurons(34),
            pval_amp=np.float64(5.5e-07),
            r_rs=np.array([0.5512674]),
            pval_rs=np.float64(0.0),
            r_of=np.array([0.7101559]),
            pval_of=np.float64(0.002),
        )
    if name == "figure2_rsof_integration":
        models = ["grs", "gof", "gratio", "gadd", "g2d"]
        model_labels = {
            "grs": "Running speed",
            "gof": "Optic flow",
            "gratio": "RS/OF",
            "gadd": "Additive",
            "g2d": "Conjunctive",
        }
        # The bootstrap p-values the STATS cell hands over. The mix is deliberate:
        # an exactly-0.0 p (what calculate_pval_from_bootstrap really returns, and
        # what the 1/n_boots floor has to catch), one already at the floor, and two
        # non-significant ones, so both _fmt_pval branches run.
        explicit = {("gadd", "g2d"): 0.155, ("gof", "gratio"): 0.0423}
        model_comparison_pvals = {}
        for i, pair in enumerate(itertools.combinations(models, 2)):
            pval = explicit.get(pair, 0.0 if i % 2 else 5e-5)
            model_comparison_pvals[pair] = {
                "pval": pval,
                "pval_floored": max(pval, 1 / 20000),
                "median_diff": 0.3,
                "ci_low": 0.2,
                "ci_high": 0.4,
            }
        return dict(
            neurons_df_sig=_neurons(12000),
            results_all=_neurons(
                25000,
                extra={
                    "iscell": 1,
                    "rsof_test_rsq_closedloop_g2d_sig": lambda n: np.arange(n) % 2 == 0,
                },
            ),
            models=models,
            model_labels=model_labels,
            model_comparison_pvals=model_comparison_pvals,
            n_boots=20000,
        )
    if name == "figure_depth_cells":
        return dict(
            n_pop_k=271,
            n_pop=285,
            n_elong=266,
            n_sessions=4,
            n_mice=4,
            k_vm=3,
            components=r"$3^\circ$ (14\%), $43^\circ$ (76\%), $90^\circ$ (11\%)",
        )
    if name == "figure_rf":
        from scipy.stats import spearmanr

        df = _neurons(
            15697,
            extra={
                "preferred_depth_closedloop": lambda n: rng.uniform(0.02, 6, n),
                "preferred_depth_corrected": lambda n: rng.uniform(0.02, 6, n),
                "overview_y_aligned": lambda n: rng.normal(size=n),
                "log_preferred_depth_corrected": lambda n: rng.normal(size=n),
            },
        )
        return dict(
            neurons_df_sig=df,
            spearmanr=spearmanr,
            p_val_corrected=7.4e-09,
            p_val_uncorrected=0.0175,
        )
    if name == "figsupp1_vis_stim_sync":
        return dict(
            desc_lag=_describe(rng.normal(26.6, 1.4, 60)),
            desc_frame_rate_2colour=_describe(rng.normal(124, 9, 25)),
            desc_frames_2colour=_describe(rng.normal(89.7, 5.1, 25)),
        )
    if name == "figsupp2_speed":
        return dict(all_data=_neurons(5000))
    if name == "figsupp4_size_control":
        return dict(
            df=_neurons(314),
            select_neurons=pd.Series(np.ones(314, dtype=bool)),
            pairwise_stats=[
                {
                    "median_ratio": 1.05,
                    "pval_ratio": 0.588,
                    "r_corr": 0.769,
                    "pval_corr": 1.8e-62,
                },
                {
                    "median_ratio": 1.00,
                    "pval_ratio": 0.812,
                    "r_corr": 0.754,
                    "pval_corr": 5.5e-59,
                },
                {
                    "median_ratio": 1.00,
                    "pval_ratio": 0.951,
                    "r_corr": 0.766,
                    "pval_corr": 8.4e-62,
                },
            ],
        )
    if name == "figsupp5_rsof":
        return dict(
            r_of_boot=np.array([-0.728]),
            pval_of_boot=0.0,
            r_rs_all_boot=np.array([0.013]),
            pval_rs_all_boot=0.715,
            bin_corrs=[0.617, 0.568, 0.489],
            bin_pvals=[0.0, 0.0, 0.0],
        )
    if name == "figsupp7_simulation_control":
        return dict(
            free={"n": 316, "n_ok": 297, "pct": 94.0},
            motor={"n": 295, "n_ok": 20, "pct": 7.0},
            n_sessions=4,
            n_mice=4,
        )
    raise KeyError(name)


# notebook stem -> the manuscript figure it must declare
OWNERS = {
    "figure1_depth_selectivity": "fig1",
    "figure_openloop": "fig2",
    "figure2_rsof_integration": "fig2",
    "figure_depth_cells": "fig3",
    "figure_rf": "fig4_5",
    "figsupp1_vis_stim_sync": "supp",
    "figsupp2_speed": "supp",
    "figsupp4_size_control": "supp",
    "figsupp5_rsof": "supp",
    "figsupp7_simulation_control": "supp",
}


def _export_cell_source(name):
    nb = json.loads((FIGURES_DIR / f"{name}.ipynb").read_text())
    cells = [
        c
        for c in nb["cells"]
        if c["cell_type"] == "code" and "export_figure_stats(" in "".join(c["source"])
    ]
    assert len(cells) == 1, f"{name}: expected 1 export cell, found {len(cells)}"
    source = "".join(cells[0]["source"])
    return source[source.index(MARKER) :]


@pytest.mark.parametrize("name,figure", sorted(OWNERS.items()))
def test_export_cell_runs_and_emits_valid_macros(name, figure):
    source = _export_cell_source(name)
    # swap the real hook for a capture so the test never writes to generated/
    source = re.sub(
        r"^from v1_depth_map\.stats import export_figure_stats$", "", source, flags=re.M
    )

    captured = {}
    namespace = _stubs(name)
    namespace.update(
        np=np,
        pd=pd,
        export_figure_stats=lambda nb_name, stats, figure=None: captured.update(
            name=nb_name, stats=stats, figure=figure
        ),
    )
    exec(compile(source, f"<{name} export cell>", "exec"), namespace)

    assert captured["name"] == name, "YAML must be keyed by the producing notebook"
    assert captured["figure"] == figure
    assert captured["stats"], "no metrics exported"

    for key, item in captured["stats"].items():
        formatted = str(item["formatted"])
        assert (
            len(UNESCAPED_DOLLAR.findall(formatted)) % 2 == 0
        ), f"{key}: {formatted!r} leaves math mode open"
        assert item["raw"] not in (None, ""), f"{key}: empty raw"
        assert re.match(r"^[A-Za-z]+$", item["latex_macro"]), f"{key}: bad macro name"
        assert item["description"], f"{key}: empty description"


def test_no_macro_is_defined_by_two_notebooks():
    """Two notebooks feed Fig. 2; they must not both claim the same macro."""
    owners = {}
    for name in OWNERS:
        tree = ast.parse(_export_cell_source(name))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.startswith("stat")
                and node.value[4:5].isupper()
            ):
                assert (
                    node.value not in owners
                ), f"macro {node.value} claimed by {owners[node.value]} and {name}"
                owners[node.value] = name
    assert len(owners) >= 40


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
