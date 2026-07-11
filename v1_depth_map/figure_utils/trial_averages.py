"""Cached trial-averaged responses for the treadmill (motor) sessions.

Several notebooks (decoder_treadmill_cut, fit_quality_viewer, state_space_geometry_pca,
session_quality_check, ...) reload the full per-frame ``trials_df`` via
``pipeline_utils.load_session`` only to reduce it to the same thing: for every closed-loop
trial, the running-masked mean running speed / optic flow and the mean dF/F of every
neuron. That reload (recording sync + dF/F load) is by far the dominant cost, and the
reduction is copy-pasted with small variations across notebooks.

This module has that reduction in one place (matching
``fit_gaussian_blob.fit_rs_of_tuning(..., trial_average=True)``) and caches the result to a
single compact pickle: one row per session, holding the per-trial ``rs`` / ``of`` vectors
and the ``dff`` matrix as arrays. The whole cache is ~10-30 MB (dff stored as float32),
versus the multi-GB per-frame ``trials_df`` it replaces.

Typical use in a notebook::

    from v1_depth_map.figure_utils import trial_averages as ta
    df = ta.load_trial_averages()                 # builds on first call, then instant
    for _, s in df.iterrows():
        C = np.column_stack([np.log(s.rs), np.log(np.degrees(s.of))])  # (logRS, logOF_deg)
        dff = s.dff                                                    # (n_trials, n_neurons)

The reduction bakes in the running mask (``rs_thr`` / ``max_rs2motor_diff``), so analyses
that need a different frame selection (e.g. the un-cut / different-mask decoders) cannot use
this cache and should keep their own path. The masking parameters are stored on every row
so a consumer can assert it matches what it expects.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.summary_analysis import get_session_list

# Kept in sync with cottage_analysis.pipelines.fit_rsof_trial_average
PROJECTS = ["ccyp_l5_3d_vision", "colasa_3d-vision_revisions"]
SESSIONS_TO_EXCLUDE = {
    "PZAG22.1b_S20260220": "1000 more frames than triggers in the treadmill recording",
}

# Canonical running mask, matching fit_rs_of_tuning(trial_average=True) as run by
# fit_rsof_trial_average.py (the source of neurons_df_trial_average.pickle).
RS_THR = 0.01
MAX_RS2MOTOR_DIFF = 0.3

# nominal depth (um) splitting Layer 2/3 from Layer 5 (decoder_treadmill_cut convention)
CUT_OFF = 350

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "trial_averages.pickle"


def _photodiode_protocol(session_name):
    return 2 if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name) else 5


def reduce_trials_to_average(
    trials_df,
    rs_thr=RS_THR,
    max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
    closed_loop_only=True,
):
    """Reduce a per-frame ``trials_df`` to per-trial averages.

    Replicates the trial-averaging path of
    ``fit_gaussian_blob.fit_rs_of_tuning(..., trial_average=True)``: for each trial, keep the
    frames where the animal is running (``RS_stim`` and ``RS_eye_stim`` above ``rs_thr``,
    finite positive ``OF_stim``, and ``max_abs_rs2motor_diff_ratio_stim`` below
    ``max_rs2motor_diff`` when that column exists), then average over surviving frames.

    Args:
        trials_df (pd.DataFrame): per-frame trials dataframe from ``pipeline_utils.load_session``.
        rs_thr (float): running-speed threshold (m/s).
        max_rs2motor_diff (float): max allowed |RS - motor| ratio per frame; ignored if the
            column is absent.
        closed_loop_only (bool): keep only ``closed_loop == 1`` trials.

    Returns:
        dict or None: ``dict(rs, of, depth_label, dff)`` where ``rs`` (m/s) and ``of`` (rad/s)
            are 1-D arrays of length ``n_trials``, ``depth_label`` is the per-trial depth label,
            and ``dff`` is a ``(n_trials, n_neurons)`` float32 array. ``None`` if no trial has
            any running frame.
    """
    if closed_loop_only and "closed_loop" in trials_df.columns:
        trials_df = trials_df[trials_df.closed_loop == 1]
    has_diff = "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns
    has_depth = "depth_labels" in trials_df.columns

    rs_l, of_l, dff_l, depth_l = [], [], [], []
    for _, tr in trials_df.iterrows():
        rs = np.asarray(tr["RS_stim"], dtype=float)
        rs_eye = np.asarray(tr["RS_eye_stim"], dtype=float)
        of = np.asarray(tr["OF_stim"], dtype=float)
        dff = np.asarray(tr["dff_stim"], dtype=float)
        run = (rs > rs_thr) & (rs_eye > rs_thr) & (~np.isnan(of)) & (of > 0)
        if has_diff:
            run = run & (
                np.asarray(tr["max_abs_rs2motor_diff_ratio_stim"], float)
                < max_rs2motor_diff
            )
        if np.sum(run) == 0:
            continue
        rs_l.append(np.mean(rs[run]))
        of_l.append(np.mean(of[run]))
        dff_l.append(np.mean(dff[run, :], axis=0))
        depth_l.append(tr["depth_labels"][0] if has_depth else np.nan)

    if not rs_l:
        return None
    return dict(
        rs=np.asarray(rs_l),
        of=np.asarray(of_l),
        depth_label=np.asarray(depth_l),
        dff=np.vstack(dff_l).astype(np.float32),
    )


def trial_average_session(
    project,
    session_name,
    rs_thr=RS_THR,
    max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
    drop_multidepth=True,
):
    """Load one session and reduce it to per-trial averages (see ``reduce_trials_to_average``).

    Returns a dict with the reduction plus ``roi`` (0..n_neurons-1, matching the dff column
    order and the ``roi`` column of neurons_df), or ``None`` if the session yields no usable
    trials.
    """
    _, _, _, trials_df = pipeline_utils.load_session(
        project=project,
        session_name=session_name,
        photodiode_protocol=_photodiode_protocol(session_name),
        regenerate_frames=False,
        filter_datasets=dict(annotated=True),
        protocol_base="SpheresTubeMotor",
    )
    if drop_multidepth:
        trials_df = trials_df[~trials_df.recording_name.str.contains("multidepth")]
    out = reduce_trials_to_average(
        trials_df, rs_thr=rs_thr, max_rs2motor_diff=max_rs2motor_diff
    )
    if out is None:
        return None
    out["roi"] = np.arange(out["dff"].shape[1])
    return out


def build_trial_average_cache(
    projects=PROJECTS,
    cache_path=DEFAULT_CACHE,
    sessions_to_exclude=SESSIONS_TO_EXCLUDE,
    rs_thr=RS_THR,
    max_rs2motor_diff=MAX_RS2MOTOR_DIFF,
    cut_off=CUT_OFF,
):
    """Build the trial-average cache for every motor session and save it as one pickle.

    The result is a DataFrame with one row per session and array-valued columns
    (``rs``, ``of``, ``depth_label``, ``dff``, ``roi``) plus scalar metadata
    (``project``, ``session``, ``nominal_depth``, ``layer``, ``n_trials``, ``n_neurons``,
    ``rs_thr``, ``max_rs2motor_diff``). Returns the DataFrame.
    """
    rows = []
    for project in projects:
        flm = flz.get_flexilims_session(project_id=project)
        sessions = get_session_list.get_motor_session_list(
            flexilims_session=flm, exclude_sessions=list(sessions_to_exclude)
        )
        entities = flz.get_entities("session", flexilims_session=flm)
        for session_name in sessions:
            if (
                session_name in sessions_to_exclude
                or session_name not in entities.index
            ):
                continue
            nominal_depth = entities.loc[session_name, "nominal_depth"]
            if isinstance(nominal_depth, (list, np.ndarray, pd.Series)):
                nominal_depth = float(np.mean(nominal_depth))
            try:
                out = trial_average_session(
                    project,
                    session_name,
                    rs_thr=rs_thr,
                    max_rs2motor_diff=max_rs2motor_diff,
                )
            except Exception as e:  # noqa: BLE001 - report and move on so one bad session
                print(f"  skip {session_name}: {e}")  # doesn't abort the whole build
                continue
            if out is None:
                print(f"  skip {session_name}: no usable trials")
                continue
            rows.append(
                dict(
                    project=project,
                    session=session_name,
                    nominal_depth=nominal_depth,
                    layer="L2/3" if nominal_depth <= cut_off else "L5",
                    n_trials=out["dff"].shape[0],
                    n_neurons=out["dff"].shape[1],
                    rs_thr=rs_thr,
                    max_rs2motor_diff=max_rs2motor_diff,
                    **out,
                )
            )
            print(
                f"  {session_name} ({project}): {out['dff'].shape[0]} trials, "
                f"{out['dff'].shape[1]} neurons"
            )

    df = pd.DataFrame(rows)
    cache_path = Path(cache_path)
    df.to_pickle(cache_path)
    total_mb = cache_path.stat().st_size / 1e6
    print(f"\nSaved {len(df)} sessions to {cache_path} ({total_mb:.1f} MB)")
    return df


def load_trial_averages(cache_path=DEFAULT_CACHE, rebuild=False, **build_kwargs):
    """Load the cached trial averages, building them on first use.

    Args:
        cache_path: pickle location (default: repo-root ``trial_averages.pickle``).
        rebuild: force a rebuild even if the cache exists.
        **build_kwargs: forwarded to ``build_trial_average_cache`` when building.

    Returns:
        pd.DataFrame: one row per session (see ``build_trial_average_cache``).
    """
    cache_path = Path(cache_path)
    if rebuild or not cache_path.exists():
        return build_trial_average_cache(cache_path=cache_path, **build_kwargs)
    return pd.read_pickle(cache_path)
