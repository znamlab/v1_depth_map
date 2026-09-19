"""Typeset every generated macro with a real LaTeX run.

The compiler's balanced-`$` and package checks are textual; this is the ground truth:
if `manuscript_stats.tex` compiles here with no packages loaded, it cannot break the
manuscript build. Skipped when pdflatex is not installed.
"""

import shutil
import subprocess
import sys
import re
from pathlib import Path

import pytest

STATS_DIR = Path(__file__).resolve().parents[1]
TEX = STATS_DIR / "manuscript_stats.tex"

pytestmark = pytest.mark.skipif(
    shutil.which("pdflatex") is None, reason="pdflatex not installed"
)


def test_every_macro_typesets(tmp_path):
    assert TEX.exists(), f"{TEX} not built yet - run build_manuscript_stats first"
    macros = re.findall(r"\\providecommand\{\\(stat[A-Za-z]+)\}", TEX.read_text())
    assert macros, "no macros in manuscript_stats.tex"

    shutil.copy2(TEX, tmp_path / TEX.name)
    body = [
        r"\documentclass{article}",
        # deliberately no packages: the macros must stand on their own
        r"\input{manuscript_stats.tex}",
        r"\begin{document}",
    ]
    for macro in macros:
        body.append("\\" + macro + "{} \\par")  # text mode
        body.append("$x$ \\" + macro + "{} $y$ \\par")  # beside math mode
    body.append(r"\end{document}")
    (tmp_path / "smoke.tex").write_text("\n".join(body) + "\n")

    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "smoke.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log = (tmp_path / "smoke.log").read_text(errors="replace")
        errors = "\n".join(re.findall(r"^!.*(?:\n.*){0,3}", log, re.M)[:5])
        pytest.fail(f"pdflatex failed on {len(macros)} macros:\n{errors}")
    assert (tmp_path / "smoke.pdf").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
