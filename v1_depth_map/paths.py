"""Centralised version and path configuration for v1_depth_map.

To bump a version across every notebook and script, change only the constant
below - there is no need to touch any notebook or precompute script.

    FIGURES_VERSION - used by figures/ *and* revisions/ notebooks and precompute
                      scripts (make_depth_tuning_raster.py, calculate_rs_stats.py)
"""

# ── Edit this line to change the active version everywhere ──────────────────
FIGURES_VERSION: str = "_rev1"  # e.g. "_rev1", "_rev2", "10"
# ────────────────────────────────────────────────────────────────────────────

# Internal directory constants - rarely need to change
_FIGURES_SUBDIR = "v1_manuscript_figures"  # used by figure notebooks


def get_figures_roots(
    flexilims_session, read_version=None, write_version=None, fig_subdir=None
):
    """Return (READ_ROOT, SAVE_ROOT) Path objects for figures notebooks.

    Args:
        flexilims_session: A flexiznam session object.
        read_version: Override the read version (defaults to FIGURES_VERSION).
        write_version: Override the write/save version (defaults to FIGURES_VERSION).
        fig_subdir: Optional subdirectory appended to both roots (e.g. "fig1",
            "fig_size_control"). Pass None to omit.

    Returns:
        tuple[Path, Path]: (READ_ROOT, SAVE_ROOT)

    Example::

        from v1_depth_map.paths import get_figures_roots
        READ_ROOT, SAVE_ROOT = get_figures_roots(flexilims_session)
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    """
    import flexiznam as flz

    rv = read_version if read_version is not None else FIGURES_VERSION
    wv = write_version if write_version is not None else FIGURES_VERSION
    base = flz.get_data_root("processed", flexilims_session=flexilims_session)
    read_root = base / _FIGURES_SUBDIR / f"ver{rv}"
    save_root = base / _FIGURES_SUBDIR / f"ver{wv}"
    if fig_subdir is not None:
        read_root = read_root / fig_subdir
        save_root = save_root / fig_subdir
    return read_root, save_root
