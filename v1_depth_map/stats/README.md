# Manuscript Statistics Tracking (`v1_depth_map.stats`)

This directory contains the pipeline for tracking and formatting manuscript numbers as LaTeX macros.


## Quick Reference
- **Detailed Documentation**: See [`notes/05_manuscript_statistics.md`](../../notes/05_manuscript_statistics.md) for full architecture, design principles, and guidelines for adding new metrics.
- **Audit Dashboard**: [`MANUSCRIPT_STATS.md`](./MANUSCRIPT_STATS.md) contains a human-readable table of all current manuscript values and macro mappings, with a `Source` column marking any value not written by a notebook run.
- **LaTeX Definitions**: [`manuscript_stats.tex`](./manuscript_stats.tex) contains the `\providecommand` and `\renewcommand` declarations.

## Commands
- **Recompile Dashboard & LaTeX macros (< 1s)**:
  ```bash
  python -m v1_depth_map.stats.build_manuscript_stats
  ```
- **Validate only, write nothing (exits non-zero on failure)**:
  ```bash
  python -m v1_depth_map.stats.build_manuscript_stats --check
  ```
- **Tests** (no 2P data needed, seconds):
  ```bash
  pytest v1_depth_map/stats/tests/
  ```
- **In Jupyter Notebooks**:
  ```python
  from v1_depth_map.stats import export_figure_stats

  export_figure_stats("figure2_openloop", stats_dict, figure="fig2")
  ```
  The first argument is the notebook's own file stem, so each notebook owns one YAML
  file and two notebooks feeding the same figure never overwrite each other. `figure`
  only groups the metrics in the dashboard and the `.tex`.
