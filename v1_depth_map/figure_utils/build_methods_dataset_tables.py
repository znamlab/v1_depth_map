"""Build summary and detailed dataset tables for the manuscript Methods section.

Constructs:
  mouse × indicator × experiment × sessions × neurons

Provides dedicated tables for:
  - closed loop
  - treadmill (motorized wheel)
  - openloop
  - sftf (drifting gratings)
  - multiday tracking
  - multidepth RF

Every number is reloaded from data:
  - closed loop / open loop / SF/TF: the cached population files of Fig 1, Fig 2 and
    Fig S10 under the figures READ_ROOT (iscell only);
  - treadmill / multi-day / multi-depth: sessions from `revisions.revision_sessions`,
    neurons counted as curated cells (`iscell` of the annotated suite2p ROIs);
  - sex: flexilims mouse entities. Indicator and genotype follow the colony prefix.

Used by `methods_dataset_tables.ipynb`, which displays the tables and prints the LaTeX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import flexiznam as flz
from cottage_analysis.io_module import suite2p as s2p_io
from v1_depth_map.revisions.revision_sessions import sessions as REVISION_SESSIONS
from v1_depth_map.revisions.revision_sessions import (
    EXCLUDED_SESSIONS as NOT_V1_SESSIONS,
)

PROJECTS = {
    "Original": "hey2_3d-vision_foodres_20220101",
    "Revisions": "colasa_3d-vision_revisions",
}

# Indicator and transgenic line, set by the colony prefix of the mouse name
LINES = {
    "PZAG": {
        "indicator": "GCaMP6s",
        "genotype": r"CamKII-tTA $\times$ TRE-GCaMP6s (G6s2)",
        "genotype_plain": "CamKII-tTA x TRE-GCaMP6s (G6s2)",
    },
    "PZAH": {
        "indicator": "GCaMP6f",
        "genotype": r"Emx1-Cre $\times$ Ai95D",
        "genotype_plain": "Emx1-Cre x Ai95D",
    },
}

EXPERIMENT_LABELS = {
    "closed_loop": "Closed loop",
    "treadmill": "Treadmill (motorized)",
    "openloop": "Open loop",
    "sftf": "SF/TF (drifting gratings)",
    "multiday": "Multi-day tracking",
    "multidepth": "Multi-depth RF",
}

# Spelled-out counts for numbers that start a sentence
_NUMBER_WORDS = {
    n + 1: w
    for n, w in enumerate(
        "One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen "
        "Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen Twenty".split()
    )
}

# Day 1 of PZAH17.1e crashed and sync between vis stim and recording is broken
# (also dropped from the multi-day panels of Fig 1)
# PZAG16.3b_S20250401 (treadmill) was not in V1 (`revision_sessions.EXCLUDED_SESSIONS`)
EXCLUDED_SESSIONS = ["PZAH17.1e_S20250304", *NOT_V1_SESSIONS]


def _revision_experiment_key(protocol: str) -> str:
    if protocol == "motor":
        return "treadmill"
    if protocol.startswith("spheretube"):
        return "multiday"
    if protocol == "multidepth":
        return "multidepth"
    raise ValueError(f"Unknown revision protocol {protocol}")


def sex_string(sexes: pd.Series, long: bool = False) -> str:
    """Format a series of "Male"/"Female" as e.g. "5M, 6F" (or "5 male, 6 female")."""
    n_m = int((sexes == "Male").sum())
    n_f = int((sexes == "Female").sum())
    m_str, f_str = (f"{n_m} male", f"{n_f} female") if long else (f"{n_m}M", f"{n_f}F")
    return ", ".join([s for n, s in [(n_m, m_str), (n_f, f_str)] if n > 0])


def get_flexilims_sessions() -> Dict[str, Any]:
    """Flexilims session of each cohort's project."""
    return {
        cohort: flz.get_flexilims_session(project)
        for cohort, project in PROJECTS.items()
    }


def load_mouse_metadata(mice, flexilims_sessions: Dict[str, Any]) -> pd.DataFrame:
    """Sex and cohort from flexilims, indicator and genotype from the colony prefix.

    Returns a DataFrame indexed by mouse name.
    """
    rows = []
    for cohort, flm_sess in flexilims_sessions.items():
        mice_ents = flz.get_entities(datatype="mouse", flexilims_session=flm_sess)
        for mouse in mice_ents.name[mice_ents.name.isin(mice)]:
            rows.append(
                {
                    "mouse": mouse,
                    "sex": mice_ents.loc[mouse, "sex"],
                    "cohort": cohort,
                    **LINES[mouse[:4]],
                }
            )
    meta = pd.DataFrame(rows).drop_duplicates("mouse").set_index("mouse")
    missing = set(mice) - set(meta.index)
    if missing:
        raise ValueError(f"Mice not found on flexilims: {sorted(missing)}")
    return meta


def count_revision_neurons(flm_sess_rev) -> pd.DataFrame:
    """Curated cells per revision session (treadmill, multi-day, multi-depth)."""
    rows = []
    for sess, protocol in REVISION_SESSIONS.items():
        if sess in EXCLUDED_SESSIONS:
            continue
        suite2p_ds = flz.get_datasets(
            origin_name=sess,
            dataset_type="suite2p_rois",
            filter_datasets={"annotated": True},
            flexilims_session=flm_sess_rev,
            allow_multiple=False,
        )
        is_cell = s2p_io.load_is_cell(suite2p_ds.path_full)
        rows.append(
            {
                "session": sess,
                "experiment_key": _revision_experiment_key(protocol),
                "neurons": int(np.sum(is_cell)),
            }
        )
    return pd.DataFrame(rows)


def _count_cached(df: pd.DataFrame, exp_key: str) -> pd.DataFrame:
    if "iscell" in df.columns:
        df = df[df["iscell"] == 1]
    counts = df.groupby("session").size().rename("neurons").reset_index()
    counts["experiment_key"] = exp_key
    return counts


def load_datasets(read_root: Path, flexilims_sessions: Dict[str, Any]) -> pd.DataFrame:
    """Per-session neuron counts across all 6 experiments, with mouse metadata."""
    counts = [
        # Closed loop (Fig 1)
        _count_cached(
            pd.read_pickle(read_root / "fig1/neurons_df_all.pickle"), "closed_loop"
        ),
        # Open loop (Fig 2)
        _count_cached(
            pd.read_pickle(read_root / "fig2/results_all_rsof_openloop.pickle"),
            "openloop",
        ),
        # SF/TF (drifting gratings, Fig S10)
        _count_cached(
            pd.read_csv(read_root / "cache_gratings/grating_tuning_cells.csv"), "sftf"
        ),
        # Treadmill (Fig 3), multi-day tracking (Fig 1N-P), multi-depth RF (Fig S8)
        count_revision_neurons(flexilims_sessions["Revisions"]),
    ]
    df = pd.concat(counts, ignore_index=True)
    df["mouse"] = df["session"].str.split("_").str[0]
    df["experiment"] = df["experiment_key"].map(EXPERIMENT_LABELS)

    meta = load_mouse_metadata(df["mouse"].unique(), flexilims_sessions)
    df = df.join(meta, on="mouse")
    df = df.rename(columns={"genotype": "genotype_tex"}).rename(
        columns={"genotype_plain": "genotype"}
    )
    return df[
        [
            "session",
            "mouse",
            "experiment_key",
            "experiment",
            "neurons",
            "indicator",
            "genotype",
            "genotype_tex",
            "sex",
            "cohort",
        ]
    ]


def mouse_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per mouse with its metadata."""
    return (
        df[["mouse", "cohort", "indicator", "genotype", "genotype_tex", "sex"]]
        .drop_duplicates("mouse")
        .set_index("mouse")
    )


def build_anonymous_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to experiment × indicator × genotype × mice × sex × sessions × neurons (no mouse names)."""
    rows = []
    for (exp_k, ind), group in df.groupby(["experiment_key", "indicator"]):
        mice_group = group[
            ["mouse", "sex", "genotype", "genotype_tex"]
        ].drop_duplicates()
        rows.append(
            {
                "experiment_key": exp_k,
                "experiment": EXPERIMENT_LABELS[exp_k],
                "indicator": ind,
                "genotype": mice_group["genotype"].iloc[0],
                "genotype_tex": mice_group["genotype_tex"].iloc[0],
                "mice": len(mice_group),
                "sex": sex_string(mice_group["sex"]),
                "sessions": group["session"].nunique(),
                "neurons": group["neurons"].sum(),
            }
        )

    exp_order = list(EXPERIMENT_LABELS.keys())
    res = pd.DataFrame(rows)
    res["exp_rank"] = res["experiment_key"].apply(exp_order.index)
    res = res.sort_values(["exp_rank", "indicator"]).drop(columns=["exp_rank"])
    return res


def build_anonymous_compact_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per experiment showing total mice, sex, indicators, sessions, and neurons (no mouse names)."""
    rows = []
    for exp_k, exp_name in EXPERIMENT_LABELS.items():
        sub = df[df["experiment_key"] == exp_k]
        if len(sub) == 0:
            continue
        mice_group = sub[["mouse", "sex", "indicator"]].drop_duplicates()
        # Most mice first, by name on ties, so the exported text is stable
        ind_counts = dict(
            sorted(
                mice_group["indicator"].value_counts().items(),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )
        rows.append(
            {
                "Experiment": exp_name,
                "Mice": len(mice_group),
                "Sex": sex_string(mice_group["sex"]),
                "Indicators (mice)": ", ".join(
                    [f"{k} ({v})" for k, v in ind_counts.items()]
                ),
                "Sessions": sub["session"].nunique(),
                "Neurons": sub["neurons"].sum(),
            }
        )
    return pd.DataFrame(rows)


def _unique_session_neurons(df: pd.DataFrame) -> pd.Series:
    """Neurons per unique session: the same cells enter several experiments (e.g. closed
    and open loop of one session), so take the largest count instead of summing."""
    return df.groupby("session")["neurons"].max()


def build_cross_experiment_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build mouse × indicator × experiment matrix of "sessions (neurons)" (Table 3).

    The last row ("Total") sums over mice.
    """
    mice = mouse_table(df)
    cohort_rank = {c: i for i, c in enumerate(PROJECTS)}
    ordered_mice = sorted(
        mice.index, key=lambda m: (cohort_rank[mice.loc[m, "cohort"]], m)
    )

    def _row(sub):
        row = {}
        for k, label in EXPERIMENT_LABELS.items():
            k_df = sub[sub["experiment_key"] == k]
            if len(k_df) > 0:
                row[label] = f"{k_df['session'].nunique()} ({k_df['neurons'].sum():,})"
            else:
                row[label] = "---"
        row["Total Sessions"] = (
            f"{sub['session'].nunique()} ({_unique_session_neurons(sub).sum():,})"
        )
        return row

    matrix_rows = []
    for m in ordered_mice:
        meta = mice.loc[m]
        matrix_rows.append(
            {
                "Mouse": m,
                "Cohort": meta["cohort"],
                "Indicator": meta["indicator"],
                "Sex": meta["sex"],
                **_row(df[df["mouse"] == m]),
            }
        )
    matrix_rows.append(
        {
            "Mouse": "Total",
            "Cohort": "",
            "Indicator": f"{len(mice)} mice",
            "Sex": sex_string(mice["sex"]),
            **_row(df),
        }
    )
    return pd.DataFrame(matrix_rows)


def generate_latex_matrix(df_mat: pd.DataFrame, use_booktabs: bool = False) -> str:
    """Generate raw LaTeX string for Table 3 with or without booktabs."""
    top_rule = r"\toprule" if use_booktabs else r"\hline"
    mid_rule = r"\midrule" if use_booktabs else r"\hline"
    bot_rule = r"\bottomrule" if use_booktabs else r"\hline"

    value_cols = ["Indicator", "Sex", *EXPERIMENT_LABELS.values(), "Total Sessions"]
    n_cols = 1 + len(value_cols)
    cols = r"@{\extracolsep{\fill}}lll" + "r" * (n_cols - 3)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}" if not use_booktabs else "",
        r"\caption{Cross-experiment summary of mice, calcium indicators, imaging sessions, and recorded layer 2/3 V1 neurons across all experimental paradigms. Values indicate the number of imaging sessions, with the total number of curated neurons shown in parentheses.}",
        r"\label{tab:methods_mouse_matrix}",
        rf"\begin{{tabular*}}{{\textwidth}}{{{cols}}}",
        top_rule,
        r"\textbf{Mouse} & \textbf{Indicator} & \textbf{Sex} & \textbf{Closed Loop} & \textbf{Treadmill} & \textbf{Open Loop} & \textbf{SF/TF} & \textbf{Multi-day} & \textbf{Multi-depth} & \textbf{Total Sessions} \\",
    ]
    lines = [line for line in lines if line]

    def _tex_row(r, bold=False):
        cells = [r["Mouse"], *[r[c] for c in value_cols]]
        if bold:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        return " & ".join(cells) + r" \\"

    mouse_rows = df_mat[df_mat["Mouse"] != "Total"]
    for cohort in PROJECTS:
        lines.append(mid_rule)
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{cohort} Cohort}}}} \\")
        for _, r in mouse_rows[mouse_rows["Cohort"] == cohort].iterrows():
            lines.append(_tex_row(r))

    lines.append(mid_rule)
    lines.append(_tex_row(df_mat[df_mat["Mouse"] == "Total"].iloc[0], bold=True))
    lines.append(bot_rule)
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_latex_compact(df_compact: pd.DataFrame, use_booktabs: bool = False) -> str:
    """Generate raw LaTeX string for the compact summary (one row per experiment)."""
    top_rule = r"\toprule" if use_booktabs else r"\hline"
    mid_rule = r"\midrule" if use_booktabs else r"\hline"
    bot_rule = r"\bottomrule" if use_booktabs else r"\hline"

    cols = ["Experiment", "Mice", "Sex", "Indicators (mice)", "Sessions", "Neurons"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.15}" if not use_booktabs else "",
        r"\caption{Summary of mice, calcium indicators, imaging sessions, and recorded layer 2/3 V1 neurons for each experimental paradigm.}",
        r"\label{tab:methods_compact_summary}",
        r"\begin{tabular}{lrllrr}",
        top_rule,
        " & ".join(rf"\textbf{{{c}}}" for c in cols) + r" \\",
        mid_rule,
    ]
    lines = [line for line in lines if line]
    for _, r in df_compact.iterrows():
        cells = [
            str(r["Experiment"]),
            str(r["Mice"]),
            str(r["Sex"]),
            str(r["Indicators (mice)"]),
            str(r["Sessions"]),
            f"{r['Neurons']:,}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(bot_rule)
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def export_dataset_table_stats(
    df_compact: pd.DataFrame,
    df_sessions: pd.DataFrame | None = None,
) -> Path:
    """Export Table S1 compact dataset summary metrics to the stats pipeline."""
    from v1_depth_map.stats import export_figure_stats

    exp_specs = {
        "Closed loop": (
            "closed_loop",
            "ClosedLoop",
            "closed loop experiments (Table S1)",
        ),
        "Treadmill (motorized)": (
            "motorized_wheel",
            "Motorized",
            "motorized wheel experiments (Table S1)",
        ),
        "Open loop": ("open_loop", "OpenLoop", "open loop experiments (Table S1)"),
        "SF/TF (drifting gratings)": (
            "drifting_gratings",
            "Gratings",
            "drifting gratings experiments (Table S1)",
        ),
        "Multi-day tracking": (
            "multiday",
            "Multiday",
            "multi-day tracking experiments (Table S1)",
        ),
        "Multi-depth RF": (
            "multidepth_rf",
            "Multidepth",
            "multi-depth RF experiments (Table S1)",
        ),
    }

    stats_data = {}
    for _, row in df_compact.iterrows():
        exp_name = row["Experiment"]
        if exp_name not in exp_specs:
            continue
        key_prefix, macro_stem, desc_label = exp_specs[exp_name]
        n_neurons = int(row["Neurons"])
        stats_data[f"{key_prefix}_neurons"] = {
            "raw": n_neurons,
            "formatted": f"{n_neurons:,}",
            "latex_macro": f"statTableSOne{macro_stem}Neurons",
            "description": f"Total recorded neurons in {desc_label}",
        }
        # Other cells of the row, exported as printed in the table
        for col, suffix, kind in [
            ("Mice", "Mice", "Mice"),
            ("Sessions", "Sessions", "Imaging sessions"),
            ("Sex", "Sex", "Sex of the mice"),
            ("Indicators (mice)", "Indicators", "Calcium indicators (number of mice)"),
        ]:
            value = row[col]
            if col in ("Mice", "Sessions"):
                value = int(value)
            stats_data[f"{key_prefix}_{suffix.lower()}"] = {
                "raw": value,
                "formatted": str(value),
                "latex_macro": f"statTableSOne{macro_stem}{suffix}",
                "description": f"{kind} in {desc_label}",
            }

    if df_sessions is not None:
        # Animals sentence of the Methods: all mice across experiments, by transgenic line
        mice = mouse_table(df_sessions)
        n_mice = len(mice)
        stats_data["total_mice"] = {
            "raw": n_mice,
            "formatted": str(n_mice),
            "latex_macro": "statTableSOneTotalMice",
            "description": "Total mice across all experiments (Methods, Animals)",
        }
        stats_data["total_mice_word"] = {
            "raw": n_mice,
            "formatted": _NUMBER_WORDS.get(n_mice, str(n_mice)),
            "latex_macro": "statTableSOneTotalMiceWord",
            "description": "Total mice, spelled out to start the Animals sentence",
        }
        for indicator, stem in [("GCaMP6f", "Emx"), ("GCaMP6s", "Gsixs")]:
            n = int((mice["indicator"] == indicator).sum())
            stats_data[f"{stem.lower()}_mice"] = {
                "raw": n,
                "formatted": str(n),
                "latex_macro": f"statTableSOne{stem}Mice",
                "description": f"Mice of the {LINES['PZAH' if indicator == 'GCaMP6f' else 'PZAG']['genotype_plain']} line ({indicator}), all experiments",
            }

        total_unique = int(_unique_session_neurons(df_sessions).sum())
        stats_data["total_recorded_neurons"] = {
            "raw": total_unique,
            "formatted": f"{total_unique:,}",
            "latex_macro": "statTableSOneTotalRecordedNeurons",
            "description": "Total unique recorded neurons across all experiments (Table S1)",
        }

    return export_figure_stats("methods_dataset_tables", stats_data, figure="supp")
