"""Location of the analysis folder (notebook caches and figure outputs).

The data package has this layout::

    <package>/raw/<project>/...
    <package>/processed/<project>/...
    <package>/analysis/            <- notebook caches and figure outputs
    <package>/offline_database.json

Everything under ``analysis/`` is disposable: delete it and rerun the notebooks with
their reload flags on to regenerate it. Notebooks and precompute scripts get this
folder only through :func:`get_figures_roots` (or :func:`get_analysis_root`) and must
not write into ``processed/`` or ``raw/``.

By default the analysis root is ``<processed root>.parent / "analysis"``, where the
processed root comes from ``flz.get_data_root("processed", ...)``. Set the environment
variable ``V1_DEPTH_MAP_ANALYSIS_ROOT`` to an absolute path to use another folder.
"""

import os
from pathlib import Path

ANALYSIS_ROOT_ENV_VAR = "V1_DEPTH_MAP_ANALYSIS_ROOT"


def get_analysis_root(flexilims_session=None):
    """Return the analysis root folder (notebook caches and figure outputs).

    Args:
        flexilims_session: A flexiznam session object, used to find the processed
            root. Not needed when ``V1_DEPTH_MAP_ANALYSIS_ROOT`` is set.

    Returns:
        Path: ``$V1_DEPTH_MAP_ANALYSIS_ROOT`` if set, otherwise
        ``<processed root>.parent / "analysis"``.
    """
    env_root = os.environ.get(ANALYSIS_ROOT_ENV_VAR)
    if env_root:
        root = Path(env_root).expanduser()
        if not root.is_absolute():
            raise ValueError(
                f"{ANALYSIS_ROOT_ENV_VAR} must be an absolute path, got {env_root!r}"
            )
        return root
    import flexiznam as flz

    processed = flz.get_data_root("processed", flexilims_session=flexilims_session)
    return Path(processed).parent / "analysis"


def get_figures_roots(
    flexilims_session, read_version=None, write_version=None, fig_subdir=None
):
    """Return (READ_ROOT, SAVE_ROOT) Path objects for figures notebooks.

    Both roots are the analysis root (see :func:`get_analysis_root`), optionally with
    ``fig_subdir`` appended. They are always the same folder.

    Args:
        flexilims_session: A flexiznam session object.
        read_version: Ignored. Kept so that older callers keep working; output
            folders are no longer versioned.
        write_version: Ignored, as ``read_version``.
        fig_subdir: Optional subdirectory appended to both roots (e.g. "fig1",
            "fig_size_control"). Pass None to omit.

    Returns:
        tuple[Path, Path]: (READ_ROOT, SAVE_ROOT)

    Example::

        from v1_depth_map.paths import get_figures_roots
        READ_ROOT, SAVE_ROOT = get_figures_roots(flexilims_session)
        SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    """
    root = get_analysis_root(flexilims_session)
    if fig_subdir is not None:
        root = root / fig_subdir
    return root, root


PER_SESSION_SUBDIR = "per_session"
PER_SESSION_CACHE_ARG = "cache_root"


def per_session_cache_kwargs(func, flexilims_session):
    """Keyword arguments sending the per-session caches of `func` to the analysis root.

    Some cottage_analysis summary functions
    (`depth_responses.get_psth_crossval_all_sessions`,
    `rs_stats.get_rs_stats_all_sessions`) cache one pickle per session. Without a
    `cache_root` argument they write it next to the session's neurons_df in
    ``processed/``. When `func` accepts `cache_root`, this returns
    ``{"cache_root": <analysis root>/per_session}`` (the function then uses
    ``<cache_root>/<session>/``); otherwise it warns and returns ``{}``.

    Args:
        func (callable): the cottage_analysis function.
        flexilims_session: A flexiznam session object.

    Returns:
        dict: keyword arguments to pass to `func`.
    """
    import inspect
    import warnings

    if PER_SESSION_CACHE_ARG in inspect.signature(func).parameters:
        root = get_analysis_root(flexilims_session) / PER_SESSION_SUBDIR
        return {PER_SESSION_CACHE_ARG: root}
    warnings.warn(
        f"{func.__module__}.{func.__qualname__} has no `{PER_SESSION_CACHE_ARG}` "
        "argument: its per-session caches are read from and written to the "
        "processed session folders",
        stacklevel=2,
    )
    return {}
