"""Tests for the manuscript statistics compiler.

Pure and fast: no 2P caches, no external drive, no notebook execution. They cover the
failure modes that would otherwise only surface as a broken Overleaf build.

Run with pytest, or standalone:
    python v1_depth_map/stats/tests/test_build_manuscript_stats.py
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from v1_depth_map.stats.build_manuscript_stats import (  # noqa: E402
    count_provenance,
    generate_latex_macros,
    generate_markdown_dashboard,
    group_by_figure,
    load_all_figure_yamls,
    export_figure_stats,
    validate,
)


def _metric(macro, formatted="42", description="a description"):
    return {
        "raw": 42,
        "formatted": formatted,
        "latex_macro": macro,
        "description": description,
    }


def _write(tmp_path, name, metrics, figure=None, source="notebook"):
    return export_figure_stats(
        name, metrics, figure=figure, output_dir=tmp_path, source=source
    )


def test_roundtrip_preserves_metrics_and_meta(tmp_path):
    _write(tmp_path, "figsupp9_v1_depth_map", {"a": _metric("statA")}, figure="supp")
    (record,) = load_all_figure_yamls(tmp_path)
    assert record["name"] == "figsupp9_v1_depth_map"
    assert record["figure"] == "supp"
    assert record["meta"]["source"] == "notebook"
    assert record["metrics"]["a"]["latex_macro"] == "statA"


def test_legacy_flat_file_still_loads(tmp_path):
    """A YAML written before the _meta/metrics split must still compile."""
    (tmp_path / "stats_fig1.yaml").write_text(yaml.dump({"a": _metric("statA")}))
    (record,) = load_all_figure_yamls(tmp_path)
    assert record["figure"] == "fig1"
    assert record["meta"]["source"] == "unknown"
    assert "a" in record["metrics"]


def test_two_notebooks_share_one_figure_section(tmp_path):
    """The whole point of keying by notebook: Fig 2 has two producers, neither clobbers."""
    _write(tmp_path, "figure2_openloop", {"a": _metric("statA")}, figure="fig2")
    _write(tmp_path, "figure2_rsof_integration", {"b": _metric("statB")}, figure="fig2")
    records = load_all_figure_yamls(tmp_path)
    assert len(records) == 2
    groups = group_by_figure(records)
    assert len(groups) == 1
    figure, group = groups[0]
    assert figure == "fig2"
    assert {r["name"] for r in group} == {
        "figure2_openloop",
        "figure2_rsof_integration",
    }


def test_figure_sections_follow_figure_order(tmp_path):
    for name, figure in [("d", "supp"), ("a", "fig1"), ("c", "fig4"), ("b", "fig2")]:
        _write(tmp_path, name, {name: _metric(f"stat{name.upper()}")}, figure=figure)
    order = [figure for figure, _ in group_by_figure(load_all_figure_yamls(tmp_path))]
    assert order == ["fig1", "fig2", "fig4", "supp"]


def test_duplicate_macro_is_an_error(tmp_path):
    _write(tmp_path, "figure2_openloop", {"a": _metric("statDup")}, figure="fig2")
    _write(tmp_path, "figsupp9_v1_depth_map", {"b": _metric("statDup")}, figure="supp")
    errors, _ = validate(load_all_figure_yamls(tmp_path))
    assert any("statDup" in e and "already defined" in e for e in errors)


def test_unbalanced_dollar_is_an_error(tmp_path):
    """The real defect: ' < 0.0001$' leaves math mode open and breaks the manuscript."""
    _write(
        tmp_path,
        "figure2_openloop",
        {"p": _metric("statP", formatted=" < 0.0001$")},
        figure="fig2",
    )
    errors, _ = validate(load_all_figure_yamls(tmp_path))
    assert any("math mode is left open" in e for e in errors)


def test_balanced_and_escaped_dollars_are_accepted(tmp_path):
    _write(
        tmp_path,
        "figure2_openloop",
        {
            "p": _metric("statP", formatted="$p < 0.0001$"),
            "pct": _metric("statPct", formatted=r"41.3\%"),
            "cost": _metric("statCost", formatted=r"\$5"),
        },
        figure="fig2",
    )
    errors, _ = validate(load_all_figure_yamls(tmp_path))
    assert errors == []


def test_package_dependent_command_is_an_error(tmp_path):
    """The macros must compile with zero external packages; \\text needs amsmath."""
    _write(
        tmp_path,
        "figsupp1_vis_stim_sync",
        {"lag": _metric("statLag", formatted=r"$26.6 \pm 1.4~\text{ms}$")},
        figure="supp",
    )
    errors, _ = validate(load_all_figure_yamls(tmp_path))
    assert any("needs the amsmath package" in e for e in errors)


def test_builtin_math_commands_are_accepted(tmp_path):
    _write(
        tmp_path,
        "figsupp1_vis_stim_sync",
        {
            "lag": _metric("statLag", formatted=r"$26.6 \pm 1.4~\mathrm{ms}$"),
            "ang": _metric("statAng", formatted=r"$3^\circ$ (14\%)"),
            "bold": _metric("statBold", formatted=r"\textbf{42}"),
        },
        figure="supp",
    )
    errors, _ = validate(load_all_figure_yamls(tmp_path))
    assert errors == []


def test_macro_name_must_be_letters_only(tmp_path):
    _write(
        tmp_path, "figsupp9_v1_depth_map", {"a": _metric("statFig4RF")}, figure="supp"
    )
    errors, _ = validate(load_all_figure_yamls(tmp_path))
    assert any("not a valid LaTeX control word" in e for e in errors)


def test_missing_description_is_only_a_warning(tmp_path):
    _write(
        tmp_path,
        "figsupp9_v1_depth_map",
        {"a": _metric("statA", description="")},
        figure="supp",
    )
    errors, warnings = validate(load_all_figure_yamls(tmp_path))
    assert errors == []
    assert any("no description" in w for w in warnings)


def test_manual_source_is_counted_and_warned(tmp_path):
    _write(tmp_path, "figsupp9_v1_depth_map", {"a": _metric("statA")}, figure="supp")
    _write(
        tmp_path,
        "figsupp2_speed",
        {"b": _metric("statB"), "c": _metric("statC")},
        figure="supp",
        source="manual",
    )
    records = load_all_figure_yamls(tmp_path)
    assert count_provenance(records) == (2, 3)
    _, warnings = validate(records)
    assert any("not notebook-derived" in w for w in warnings)


def test_stale_yaml_is_warned(tmp_path, monkeypatch=None):
    """A YAML older than its notebook means the numbers predate the current code."""
    import v1_depth_map.stats.build_manuscript_stats as bms

    notebooks = tmp_path / "figures"
    notebooks.mkdir()
    (notebooks / "figsupp9_v1_depth_map.ipynb").write_text("{}")
    original = bms.NOTEBOOKS_DIR
    bms.NOTEBOOKS_DIR = notebooks
    try:
        path = _write(
            tmp_path, "figsupp9_v1_depth_map", {"a": _metric("statA")}, figure="supp"
        )
        doc = yaml.safe_load(path.read_text())
        doc["_meta"]["generated_at"] = "2000-01-01T00:00:00"
        path.write_text(yaml.dump(doc))
        _, warnings = validate(load_all_figure_yamls(tmp_path))
        assert any("stale" in w for w in warnings)
    finally:
        bms.NOTEBOOKS_DIR = original


def test_outputs_mention_provenance(tmp_path):
    _write(
        tmp_path,
        "figsupp9_v1_depth_map",
        {"a": _metric("statA")},
        figure="supp",
        source="manual",
    )
    records = load_all_figure_yamls(tmp_path)

    md = tmp_path / "out.md"
    generate_markdown_dashboard(records, md)
    md_text = md.read_text()
    assert "1 of 1 values are hand-entered" in md_text
    assert "`\\statA`" in md_text

    tex = tmp_path / "out.tex"
    generate_latex_macros(records, tex)
    tex_text = tex.read_text()
    assert "WARNING: 1 of 1 values are hand-entered" in tex_text
    assert "\\providecommand{\\statA}{42}" in tex_text
    assert "\\renewcommand{\\statA}{42}" in tex_text


def test_all_notebook_source_reports_clean(tmp_path):
    _write(tmp_path, "figsupp9_v1_depth_map", {"a": _metric("statA")}, figure="supp")
    md = tmp_path / "out.md"
    generate_markdown_dashboard(load_all_figure_yamls(tmp_path), md)
    assert "All 1 values were written by a notebook run" in md.read_text()


if __name__ == "__main__":
    import shutil
    import tempfile
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, func in tests:
        tmp = Path(tempfile.mkdtemp())
        try:
            func(tmp)
            print(f"PASS {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
