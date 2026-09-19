"""Manuscript statistics automation package.

Numbers are emitted directly from the figure notebooks that generate the plots
to prevent drift between figures and reported text/captions.
"""

from v1_depth_map.stats.build_manuscript_stats import (
    compile_manuscript_stats,
    export_figure_stats,
)

__all__ = [
    "export_figure_stats",
    "compile_manuscript_stats",
]
