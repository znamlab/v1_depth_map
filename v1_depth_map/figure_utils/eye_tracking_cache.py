"""Build the eye-tracking caches read by `figures/figsupp2_speed.ipynb` (Fig. S2 eye panels).

The two pickles are notebook caches and live in `<analysis>/eye_tracking/`, where
`<analysis>` is the analysis root of `v1_depth_map.paths.get_analysis_root` (deleting it and
reloading rebuilds them). When they are missing there, `load_eye_tracking_cache` reads the
copies at the legacy location `<processed>/<project>/Analysis/eye_tracking/` if present, and
never writes there:

- `all_data.pkl`: one row per right-eye camera frame for every eye-tracking session,
  concatenated with a fresh RangeIndex. Columns are the wayla ellipse fits (centre, radii,
  angle, fit error, rsquare, average DLC likelihood, reflection position/likelihood, pupil
  position relative to the reflection, `phi`/`theta`/`radius` from the eye reprojection),
  the gaze in world coordinates (`azimuth`, `elevation`, in degrees, after the camera
  extrinsics), the behaviour of each frame (`is_stim`, `trial`, `depth`, `RS`, `OF`,
  `harptime`), eye-opening and validity flags, interpolated/box-filtered gaze, gaze
  displacement and velocity, `depth_label` and `session`.
- `example_dlc_res.pkl`: the raw DeepLabCut output (columns scorer/bodyparts/coords) of
  the cropped right-eye tracking of the example session.

This restores the builder cell of the 2024 notebook
`v1_depth_analysis/eye_tracking/eye_tracking_figure.ipynb` (commits 862e0d3 to a3efc2d,
later moved to `figures/figsupp_eye_tracking.ipynb`, deleted in c953293). The per-session
work is done by `v1_depth_map.batch_analysis.eye_tracking.analysis.get_data`, unchanged
since then apart from the data-root lookup and `ast_neuropil=False` in the trial sync. The
original cell compared the integer row index with the example session name, so it could
not have saved `example_dlc_res` as written; here the example session is matched by name.

Usage::

    import flexiznam as flz
    from v1_depth_map.figure_utils import eye_tracking_cache as etc

    flm_sess = flz.get_flexilims_session(project_id=etc.PROJECT)
    all_data, example_dlc_res = etc.build_eye_tracking_cache(
        flm_sess, output_dir=etc.get_cache_folder(flm_sess)
    )
"""

from pathlib import Path

import flexiznam as flz
import pandas as pd
from cottage_analysis.summary_analysis import get_session_list

from v1_depth_map.batch_analysis.eye_tracking.analysis import get_data
from v1_depth_map.paths import get_analysis_root

PROJECT = "hey2_3d-vision_foodres_20220101"
# Only these two mice have camera extrinsics (see `analysis.CALIBRATION_SESSION`)
EYE_TRACKING_MICE = ("PZAH6.4b", "PZAG3.4f")
SESSIONS_TO_EXCLUDE = {
    "PZAH6.4b_S20220516": "Mouse squint too much, cropping issue",
    "PZAH6.4b_S20220429": "Two eye position. Maybe reflection ill detected",
}
EXAMPLE_SESSION = "PZAG3.4f_S20220421"
PROTOCOL = "SpheresPermTubeReward"
FILT_WINDOW = 3
ALL_DATA_FILE = "all_data.pkl"
EXAMPLE_DLC_FILE = "example_dlc_res.pkl"
CACHE_SUBDIR = "eye_tracking"


def get_cache_folder(flexilims_session, project=PROJECT):
    """Folder holding the eye-tracking caches: `<analysis root>/eye_tracking`.

    `project` is unused and kept for backward compatibility.
    """
    return get_analysis_root(flexilims_session) / CACHE_SUBDIR


def get_legacy_cache_folder(flexilims_session, project=PROJECT):
    """Legacy (read-only) cache folder: `<processed>/<project>/Analysis/eye_tracking`."""
    processed = flz.get_data_root(
        "processed", project=project, flexilims_session=flexilims_session
    )
    return Path(processed) / project / "Analysis" / "eye_tracking"


def get_eye_tracking_sessions(flexilims_session, exclude=SESSIONS_TO_EXCLUDE):
    """Closed-loop V1 sessions of the eye-tracking mice, minus `exclude`.

    Args:
        flexilims_session (flexilims.Session): session of the main project.
        exclude (iterable, optional): session names to drop. Defaults to
            `SESSIONS_TO_EXCLUDE`.

    Returns:
        list: session names, in database order.
    """
    sessions = get_session_list.get_sessions(
        flexilims_session=flexilims_session,
        exclude_sessions=(),
        exclude_openloop=True,
        exclude_pure_closedloop=False,
        v1_only=True,
    )
    return [
        s for s in sessions if s.split("_")[0] in EYE_TRACKING_MICE and s not in exclude
    ]


def load_session_gaze(
    session_name,
    flexilims_session,
    project=PROJECT,
    filt_window=FILT_WINDOW,
    verbose=False,
):
    """Gaze, behaviour and DLC output of the first closed-loop recording of a session.

    Args:
        session_name (str): `<mouse>_<session>`.
        flexilims_session (flexilims.Session): session of `project`.
        project (str, optional): project name. Defaults to `PROJECT`.
        filt_window (int, optional): box filter on azimuth/elevation, in frames.
            Defaults to 3.
        verbose (bool, optional): print progress from `get_data`. Defaults to False.

    Returns:
        pd.DataFrame: gaze data, one row per camera frame, with a `session` column.
        pd.DataFrame: DLC tracking results (cropped video).
        flexiznam.Dataset: DLC tracking dataset.
    """
    sess_entity = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flexilims_session
    )
    if sess_entity is None:
        raise IOError(f"Session {session_name} not found in flexilims")
    recordings = flz.get_entities(
        datatype="recording",
        origin_id=sess_entity["id"],
        flexilims_session=flexilims_session,
    )
    recording = recordings[recordings.protocol == PROTOCOL].iloc[0]
    gaze_data, dlc_res, dlc_ds = get_data(
        project=project,
        mouse=sess_entity.genealogy[0],
        session=sess_entity.genealogy[-1],
        recording=recording.genealogy[-1],
        filt_window=filt_window,
        verbose=verbose,
    )
    gaze_data["session"] = session_name
    return gaze_data, dlc_res, dlc_ds


def build_all_data(
    flexilims_session,
    project=PROJECT,
    sessions=None,
    output_dir=None,
    example_session=EXAMPLE_SESSION,
    filt_window=FILT_WINDOW,
    verbose=False,
):
    """Concatenate the gaze data of all sessions, as in `all_data.pkl`.

    Sessions that fail are skipped with a message, as in the original notebook.

    Args:
        flexilims_session (flexilims.Session): session of `project`.
        project (str, optional): project name. Defaults to `PROJECT`.
        sessions (list, optional): session names. Defaults to
            `get_eye_tracking_sessions(flexilims_session)`.
        output_dir (Path, optional): if given, write `all_data.pkl` there.
        example_session (str, optional): session whose DLC output is returned.
        filt_window (int, optional): see `load_session_gaze`.
        verbose (bool, optional): see `load_session_gaze`.

    Returns:
        pd.DataFrame: concatenated gaze data.
        pd.DataFrame or None: DLC output of `example_session`, None if not in
            `sessions`.
        list: sessions that raised an error.
    """
    if sessions is None:
        sessions = get_eye_tracking_sessions(flexilims_session)
    gaze_by_session = {}
    example_dlc_res = None
    problematic_sessions = []
    for session_name in sessions:
        print(session_name, flush=True)
        try:
            gaze_data, dlc_res, _ = load_session_gaze(
                session_name,
                flexilims_session,
                project=project,
                filt_window=filt_window,
                verbose=verbose,
            )
        except Exception as err:
            print(f"Problem with {session_name}: {err}")
            problematic_sessions.append(session_name)
            continue
        gaze_by_session[session_name] = gaze_data
        if session_name == example_session:
            example_dlc_res = dlc_res
    if not gaze_by_session:
        raise IOError("No session could be loaded")
    all_data = pd.concat(gaze_by_session, names=["session"], ignore_index=True)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        all_data.to_pickle(output_dir / ALL_DATA_FILE)
    return all_data, example_dlc_res, problematic_sessions


def build_example_dlc_res(
    flexilims_session,
    project=PROJECT,
    example_session=EXAMPLE_SESSION,
    output_dir=None,
):
    """DLC output of the example session, as in `example_dlc_res.pkl`."""
    _, dlc_res, _ = load_session_gaze(
        example_session, flexilims_session, project=project
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dlc_res.to_pickle(output_dir / EXAMPLE_DLC_FILE)
    return dlc_res


def build_eye_tracking_cache(
    flexilims_session,
    output_dir,
    project=PROJECT,
    sessions=None,
    example_session=EXAMPLE_SESSION,
    filt_window=FILT_WINDOW,
    verbose=False,
):
    """Build and save `all_data.pkl` and `example_dlc_res.pkl` in `output_dir`.

    The example session is loaded once more only if it is not in `sessions`.

    Returns:
        pd.DataFrame: all_data.
        pd.DataFrame: example_dlc_res.
    """
    all_data, example_dlc_res, problematic = build_all_data(
        flexilims_session,
        project=project,
        sessions=sessions,
        output_dir=output_dir,
        example_session=example_session,
        filt_window=filt_window,
        verbose=verbose,
    )
    if problematic:
        print(f"Sessions skipped: {problematic}")
    if example_dlc_res is None:
        example_dlc_res = build_example_dlc_res(
            flexilims_session, project=project, example_session=example_session
        )
    example_dlc_res.to_pickle(Path(output_dir) / EXAMPLE_DLC_FILE)
    return all_data, example_dlc_res


def load_eye_tracking_cache(
    flexilims_session,
    project=PROJECT,
    reload=False,
    output_dir=None,
    sessions=None,
):
    """Read the eye-tracking caches, rebuilding them first if `reload` or missing.

    With the default `output_dir`, caches missing from `get_cache_folder` are read
    from `get_legacy_cache_folder` if both files are there. Rebuilt caches are always
    written to `output_dir` (default `get_cache_folder`), never to the legacy folder.

    Args:
        flexilims_session (flexilims.Session): session of `project`.
        project (str, optional): project name. Defaults to `PROJECT`.
        reload (bool, optional): rebuild from the tracking data. Defaults to False.
        output_dir (Path, optional): cache folder. Defaults to `get_cache_folder`.
        sessions (list, optional): sessions to rebuild, see `build_all_data`.

    Returns:
        pd.DataFrame: all_data.
        pd.DataFrame: example_dlc_res.
    """
    default_dir = output_dir is None
    if default_dir:
        output_dir = get_cache_folder(flexilims_session, project=project)
    output_dir = Path(output_dir)
    files = [output_dir / ALL_DATA_FILE, output_dir / EXAMPLE_DLC_FILE]
    if default_dir and not reload and not all(f.exists() for f in files):
        legacy_dir = get_legacy_cache_folder(flexilims_session, project=project)
        legacy_files = [legacy_dir / ALL_DATA_FILE, legacy_dir / EXAMPLE_DLC_FILE]
        if all(f.exists() for f in legacy_files):
            print(f"Reading eye-tracking caches from legacy folder {legacy_dir}")
            files = legacy_files
    if reload or not all(f.exists() for f in files):
        return build_eye_tracking_cache(
            flexilims_session, output_dir, project=project, sessions=sessions
        )
    return pd.read_pickle(files[0]), pd.read_pickle(files[1])
