"""Compile manuscript statistics from notebook-emitted YAML files.

The figure notebooks emit their own numbers directly during execution to guarantee
that figures, captions, and text numbers remain 100% in sync without drift.

`export_figure_stats` (called from the notebooks) writes those YAML files; the rest of
this module aggregates them and compiles:
1. `MANUSCRIPT_STATS.md`: A comprehensive, human-readable audit dashboard.
2. `manuscript_stats.tex`: LaTeX macros (\\providecommand & \\renewcommand) ready to be cited.

Both are written into this directory and nowhere else. Getting the macros into the
manuscript is a deliberate manual step: once the dashboard has been checked, copy
`manuscript_stats.tex` into the Overleaf project folder by hand. Nothing here writes to
Overleaf, so a compile can never surprise a co-author mid-edit.

One YAML file per *notebook*, not per figure: a figure can have several producing
notebooks (Fig. 2 panels A-E come from `figure2_rsof_integration`, F-L from
`figure_openloop`), and keying by figure made them overwrite each other. Each file
carries a `_meta` block recording which notebook wrote it, when, and whether the
numbers came from a notebook run or were entered by hand.
"""

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple
import yaml

STATS_DIR = Path(__file__).parent
GENERATED_DIR = STATS_DIR / "generated"
NOTEBOOKS_DIR = STATS_DIR.parent / "figures"

# Section order in the dashboard and in the .tex. Anything not listed is appended.
FIGURE_ORDER = ["fig1", "fig2", "fig3", "fig4_5", "supp"]

FIGURE_TITLES = {
    "fig1": "Figure 1: Virtual Depth Selectivity & Experience Independence",
    "fig2": "Figure 2: Visuomotor Integration Models & Open-Loop Replay",
    "fig3": "Figure 3: Motorized Wheel & Visuomotor Gain Modulation",
    "fig4_5": "Figures 4 & 5: Three-Dimensional Receptive Fields & V1 Depth Map",
    "supp": "Supplementary Figures & Methods: Controls & Experimental Specs",
}

FIGURE_TITLES_TEX = {
    "fig1": "Figure 1: Depth Selectivity & Stability",
    "fig2": "Figure 2: Visuomotor Integration Models & Open Loop",
    "fig3": "Figure 3: Motorized Wheel & Gain Modulation",
    "fig4_5": "Figures 4 & 5: 3D Receptive Fields & V1 Depth Map",
    "supp": "Supplementary Figures & Controls",
}

MACRO_RE = re.compile(r"^[A-Za-z]+$")
UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")

# Design rule: the macros must compile with zero external packages, so that Overleaf
# and the journal's conversion tools can use them as-is. These commands do not exist
# in plain LaTeX; each maps to a built-in that renders the same way.
PACKAGE_COMMANDS = {
    r"\text": (r"\mathrm", "amsmath"),
    r"\bm": (r"\mathbf", "bm"),
    r"\SI": ("a plain number and unit", "siunitx"),
    r"\num": ("a plain number", "siunitx"),
    r"\degree": (r"^\circ", "gensymb"),
    r"\nicefrac": (r"\frac", "nicefrac"),
}
PACKAGE_COMMAND_RE = re.compile(
    "(" + "|".join(re.escape(c) for c in PACKAGE_COMMANDS) + r")(?![A-Za-z])"
)


def export_figure_stats(
    name: str,
    stats_data: Dict[str, Any],
    figure: str = None,
    output_dir: Path = GENERATED_DIR,
    source: str = "notebook",
) -> Path:
    """Write the statistics a notebook computed for its panels to a human-readable YAML.

    Called at the end of a figure notebook:

        from v1_depth_map.stats import export_figure_stats
        export_figure_stats("figure_openloop", stats_dict, figure="fig2")

    `name` identifies the producing notebook (its stem) and decides the file name, so two
    notebooks feeding the same manuscript figure never overwrite each other. `figure` is
    the manuscript figure the metrics belong to and only decides how they are grouped in
    the dashboard and the .tex. `source` records provenance: anything other than
    "notebook" is counted and flagged as not notebook-derived.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / f"stats_{name}.yaml"
    document = {
        "_meta": {
            "notebook": name,
            "figure": figure or name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source": source,
        },
        "metrics": stats_data,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            document,
            f,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"[Stats] Saved {len(stats_data)} metrics for {name} to: {yaml_path.name}")
    return yaml_path


def load_all_figure_yamls(yaml_dir: Path = GENERATED_DIR) -> List[Dict[str, Any]]:
    """Load every stats_*.yaml emitted by the figure notebooks.

    Returns one record per file: `{"name", "path", "figure", "meta", "metrics"}`.
    Files written before the `_meta`/`metrics` split are read as flat metric maps and
    grouped by their file stem, so an un-migrated YAML still compiles.
    """
    records = []
    for path in sorted(yaml_dir.glob("stats_*.yaml")):
        name = path.stem.replace("stats_", "")
        with open(path, "r", encoding="utf-8") as f:
            document = yaml.safe_load(f) or {}

        if "metrics" in document or "_meta" in document:
            meta = document.get("_meta", {}) or {}
            metrics = document.get("metrics", {}) or {}
        else:  # legacy flat file: the whole document is the metric map
            meta = {"notebook": name, "figure": name, "source": "unknown"}
            metrics = document

        records.append(
            {
                "name": name,
                "path": path,
                "figure": meta.get("figure") or name,
                "meta": meta,
                "metrics": metrics,
            }
        )
    return records


def group_by_figure(
    records: List[Dict[str, Any]],
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Group notebook records by manuscript figure, in FIGURE_ORDER then alphabetically."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["figure"], []).append(record)

    def sort_key(figure: str):
        if figure in FIGURE_ORDER:
            return (0, FIGURE_ORDER.index(figure), "")
        return (1, 0, figure)

    return [(figure, groups[figure]) for figure in sorted(groups, key=sort_key)]


def validate(records: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Check every metric before anything is written. Returns (errors, warnings).

    These are the failures that would otherwise only show up as a broken Overleaf
    build: a macro defined twice with different values, a `formatted` string that
    leaves math mode open, or a macro name LaTeX cannot define.
    """
    errors: List[str] = []
    warnings: List[str] = []
    seen_macros: Dict[str, str] = {}

    for record in records:
        origin = record["path"].name
        if not record["metrics"]:
            warnings.append(f"{origin}: no metrics in file")

        for key, item in record["metrics"].items():
            if not isinstance(item, dict):
                errors.append(f"{origin}:{key}: entry is not a mapping")
                continue

            macro = item.get("latex_macro", key)
            if not MACRO_RE.match(str(macro)):
                errors.append(
                    f"{origin}:{key}: macro name '{macro}' is not a valid LaTeX "
                    "control word (letters only)"
                )
            elif macro in seen_macros:
                errors.append(
                    f"{origin}:{key}: macro '\\{macro}' already defined in {seen_macros[macro]}"
                )
            else:
                seen_macros[macro] = origin

            formatted = item.get("formatted", item.get("raw", ""))
            n_dollars = len(UNESCAPED_DOLLAR_RE.findall(str(formatted)))
            if n_dollars % 2:
                errors.append(
                    f"{origin}:{key}: formatted value {formatted!r} has {n_dollars} "
                    "unescaped '$' - math mode is left open"
                )

            for command in set(PACKAGE_COMMAND_RE.findall(str(formatted))):
                replacement, package = PACKAGE_COMMANDS[command]
                errors.append(
                    f"{origin}:{key}: formatted value uses '{command}', which needs the "
                    f"{package} package - use {replacement} instead"
                )

            if not item.get("description"):
                warnings.append(f"{origin}:{key}: no description")

        warnings.extend(_staleness_warnings(record))

    return errors, warnings


def _staleness_warnings(record: Dict[str, Any]) -> List[str]:
    """Warn when a YAML predates the notebook that owns it, or was hand-entered."""
    warnings = []
    meta = record["meta"]
    source = meta.get("source", "unknown")

    if source != "notebook":
        warnings.append(
            f"{record['path'].name}: source is '{source}' - "
            f"{len(record['metrics'])} value(s) are not notebook-derived"
        )
        return warnings

    notebook = NOTEBOOKS_DIR / f"{meta.get('notebook', record['name'])}.ipynb"
    generated_at = meta.get("generated_at")
    if not notebook.exists() or not generated_at:
        return warnings
    try:
        stamp = datetime.fromisoformat(str(generated_at))
    except ValueError:
        return warnings
    if stamp.timestamp() < notebook.stat().st_mtime:
        warnings.append(
            f"{record['path'].name}: stale - {notebook.name} was modified after these "
            f"numbers were generated ({generated_at})"
        )
    return warnings


def count_provenance(records: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Return (n_metrics_not_from_a_notebook_run, n_metrics_total)."""
    total = sum(len(r["metrics"]) for r in records)
    manual = sum(
        len(r["metrics"])
        for r in records
        if r["meta"].get("source", "unknown") != "notebook"
    )
    return manual, total


def generate_markdown_dashboard(
    records: List[Dict[str, Any]], output_path: Path
) -> None:
    """Generate a clean Markdown audit dashboard from the notebook-emitted statistics."""
    manual, total = count_provenance(records)

    lines = [
        "# Manuscript Statistics Manifest (Audit Dashboard)",
        "",
        f"*Compiled from figure notebook outputs at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.*",
        "",
        "This document is compiled directly from statistics generated inside the figure notebooks. ",
        "Because each number is written by the exact code cell that generates the corresponding figure panel, ",
        "these values cannot drift out of sync with the figures.",
        "",
    ]

    if manual:
        lines.extend(
            [
                f"> ⚠️ **{manual} of {total} values are hand-entered, not notebook-derived.** ",
                "> They are marked `manual` in the Source column below and are provisional until the ",
                "> owning notebook is executed and overwrites them. ",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"> ✅ All {total} values were written by a notebook run.",
                "",
            ]
        )

    lines.extend(["---", ""])

    for figure, group in group_by_figure(records):
        lines.append(f"## {FIGURE_TITLES.get(figure, f'Figure: {figure.upper()}')}")
        lines.append("")
        for record in sorted(group, key=lambda r: r["name"]):
            meta = record["meta"]
            source = meta.get("source", "unknown")
            lines.append(
                f"Source notebook: `{meta.get('notebook', record['name'])}.ipynb` "
                f"({source}, generated {meta.get('generated_at', 'unknown')})"
            )
            lines.append("")
            lines.append(
                "| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |"
            )
            lines.append("| :--- | :---: | :---: | :---: | :--- |")
            for key, item in record["metrics"].items():
                macro = f"`\\{item.get('latex_macro', key)}`"
                formatted = f"`{item.get('formatted', '')}`"
                raw = str(item.get("raw", ""))
                desc = item.get("description", "")
                lines.append(f"| {macro} | {formatted} | {raw} | {source} | {desc} |")
            lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Usage in LaTeX (`v1_depth_map.tex`)",
            "",
            "To cite any statistic in the LaTeX manuscript, include `\\input{manuscript_stats.tex}` in the preamble, ",
            "then use the macro with empty braces to preserve following spaces: ",
            "",
            "```latex",
            "\\input{manuscript_stats.tex}",
            "",
            "% In text:",
            "Across the population, \\statFigOnePctDepthNeurons{} of cells ",
            "(\\statFigOneDepthNeurons{} of \\statFigOneTotalNeurons{} neurons) exhibited significant depth selectivity...",
            "```",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_latex_macros(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Generate manuscript_stats.tex with \\newcommand declarations for each statistic."""
    manual, total = count_provenance(records)

    lines = [
        "% ==============================================================================",
        "% Auto-generated from figure notebook outputs - DO NOT EDIT MANUALLY",
        f"% Compiled at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if manual:
        lines.append(
            f"% WARNING: {manual} of {total} values are hand-entered, not notebook-derived."
        )
        lines.append("%          See MANUSCRIPT_STATS.md for which ones.")
    else:
        lines.append(f"% All {total} values were written by a notebook run.")
    lines.extend(
        [
            "% ==============================================================================",
            "",
        ]
    )

    for figure, group in group_by_figure(records):
        lines.append(f"% --- {FIGURE_TITLES_TEX.get(figure, figure.upper())} ---")
        for record in sorted(group, key=lambda r: r["name"]):
            meta = record["meta"]
            lines.append(
                f"% from {meta.get('notebook', record['name'])}.ipynb "
                f"({meta.get('source', 'unknown')})"
            )
            for key, item in record["metrics"].items():
                macro_name = item.get("latex_macro", key)
                val = item.get("formatted", str(item.get("raw", "")))
                desc = item.get("description", "")
                if desc:
                    lines.append(f"% {desc}")
                lines.append(f"\\providecommand{{\\{macro_name}}}{{{val}}}")
                lines.append(f"\\renewcommand{{\\{macro_name}}}{{{val}}}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compile_manuscript_stats(
    yaml_dir: Path = GENERATED_DIR, check_only: bool = False
) -> int:
    """Compile all notebook-generated YAML files into Markdown dashboard and LaTeX macros.

    Returns 0 on success, 1 if validation failed (nothing is written in that case).
    """
    print("=" * 70)
    print("Manuscript Statistics Compilation (From Notebook YAML Outputs)")
    print("=" * 70)

    records = load_all_figure_yamls(yaml_dir)
    manual, total = count_provenance(records)
    print(
        f"Loaded {len(records)} notebook stat files ({total} metrics) from {yaml_dir.name}/"
    )

    errors, warnings = validate(records)
    for warning in warnings:
        print(f"[Warning] {warning}")
    if errors:
        for error in errors:
            print(f"[Error]   {error}")
        print("=" * 70)
        print(f"Validation FAILED with {len(errors)} error(s). Nothing was written.")
        print("=" * 70)
        return 1

    if check_only:
        print("=" * 70)
        print(
            f"Validation passed ({total} metrics, {manual} hand-entered). Nothing written."
        )
        print("=" * 70)
        return 0

    md_path = STATS_DIR / "MANUSCRIPT_STATS.md"
    generate_markdown_dashboard(records, md_path)
    print(f"[Compiled] Markdown Dashboard: {md_path}")

    tex_path = STATS_DIR / "manuscript_stats.tex"
    generate_latex_macros(records, tex_path)
    print(f"[Compiled] LaTeX Macros: {tex_path}")

    if manual:
        print(
            f"[Notice] {manual} of {total} values are hand-entered, not notebook-derived."
        )

    print("=" * 70)
    print("Compilation completed successfully!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile manuscript statistics from notebook YAMLs."
    )
    parser.add_argument(
        "--yaml-dir",
        type=str,
        default=str(GENERATED_DIR),
        help="Directory containing stats_*.yaml files emitted by notebooks.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the YAML files and exit non-zero on failure, without writing anything.",
    )
    args = parser.parse_args()
    sys.exit(
        compile_manuscript_stats(yaml_dir=Path(args.yaml_dir), check_only=args.check)
    )
