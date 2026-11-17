"""RF fits of the treadmill (SpheresTubeMotor) recordings of one session.

The analysis pipeline skips the RF step for SpheresTubeMotor. This runs the same RF step
(`analysis_pipeline.fit_session_rfs`: same stimulus, lag, hyperparameter grid, ipsi fit
and RF depth fit) on the treadmill recordings. The RF columns get a `_treadmill` suffix
(`rf_coef_closedloop_treadmill`, ...). The other columns are copied from
`neurons_df.pickle`, which is never modified; the result is saved next to it as
`neurons_df_rf_only_treadmill.pickle` (`..._treadmill_spks.pickle` for spikes).

The treadmill protocol spans 2.2 mm to 36.7 m, beyond the default RF depth-fit bounds
of `fit_rf_preferred_depth` (2 cm to 20 m, set for the closed-loop depths), which makes
the fit fail for ROIs whose RF peaks at an extreme depth. The bounds are set from the
session depths instead, as in the treadmill depth-tuning fit: log(round(min, 4)) to
log(ceil(max)).

Usage:
    python rf_fit_treadmill.py SESSION --project PROJECT [--photodiode 5] [--annotated]
    python rf_fit_treadmill.py SESSION --project PROJECT --annotated --refit-depth-only
        (redo only the RF preferred-depth columns of an existing output file)
"""

import argparse
import functools
import json
import os

import numpy as np
import pandas as pd
import flexiznam as flz

from cottage_analysis.analysis import find_depth_neurons, treadmill
from cottage_analysis.analysis.spheres import rf_analysis
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.pipelines.analysis_pipeline import fit_session_rfs

PROTOCOL_BASE = "SpheresTubeMotor"
SUFFIX = "_treadmill"


def depth_bounds_from(depth_list):
    """log-depth bounds of the treadmill depth-tuning fit (refit_treadmill_depth)."""
    depth_list = np.asarray(depth_list, dtype=float)
    return (np.log(np.round(depth_list.min(), 4)), np.log(np.ceil(depth_list.max())))


def refit_rf_depth(neurons_df, depth_list):
    """Refit the `rf_*depth*_closedloop_treadmill` columns from the stored coefficients."""
    scratch = pd.DataFrame(
        {"rf_coef_closedloop": neurons_df[f"rf_coef_closedloop{SUFFIX}"].values}
    )
    rf_analysis.fit_rf_preferred_depth(
        scratch,
        depths=depth_list,
        is_closed_loop=1,
        depth_bounds=depth_bounds_from(depth_list),
    )
    for col in ["rf_preferred_depth", "rf_depth_popt", "rf_depth_rsq"]:
        neurons_df[f"{col}_closedloop{SUFFIX}"] = scratch[f"{col}_closedloop"].values
    return neurons_df


def main(
    session,
    project,
    photodiode_protocol,
    annotated,
    use_col="dffs",
    refit_depth_only=False,
):
    fs = flz.get_flexilims_session(project)
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=fs, project=project, conflicts="skip"
    )
    src = neurons_ds.path_full
    out_sfx = "" if use_col == "dffs" else f"_{use_col}"
    target = src.with_name(f"neurons_df_rf_only{SUFFIX}{out_sfx}.pickle")
    neurons_df = pd.read_pickle(src)
    print(f"{session}: {len(neurons_df)} ROIs from {src}", flush=True)
    filter_traces = {"anatomical_only": 3, "ast_neuropil": False}
    if annotated:
        filter_traces["annotated"] = True

    _, trials_df_all = treadmill.sync_all_recordings(
        session_name=session,
        flexilims_session=fs,
        project=project,
        filter_datasets=filter_traces,
        conflicts="skip",
        recording_type="two_photon",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
        method="plateau",
    )
    if "closed_loop" not in trials_df_all.columns:
        trials_df_all["closed_loop"] = 1
    nrois = trials_df_all.dff_stim.iloc[0].shape[1]
    assert nrois == len(
        neurons_df
    ), f"{nrois} ROIs in traces, {len(neurons_df)} in neurons_df"
    depth_list = find_depth_neurons.find_depth_list(trials_df_all)
    bounds = depth_bounds_from(depth_list)
    print(
        f"{len(depth_list)} depths {np.round(depth_list, 4)}; RF depth bounds {np.exp(bounds)}"
    )

    if refit_depth_only:
        out = pd.read_pickle(target)
        assert np.array_equal(
            out.roi.values, neurons_df.roi.values
        ), "ROI order changed"
        out = refit_rf_depth(out, depth_list)
        tmp = target.with_suffix(".tmp.pickle")
        out.to_pickle(tmp)
        os.replace(tmp, target)
        meta = (
            json.loads(target.with_suffix(".json").read_text())
            if target.with_suffix(".json").exists()
            else {}
        )
        meta.update(
            depths=list(map(float, depth_list)),
            rf_depth_bounds=list(map(float, np.exp(bounds))),
        )
        target.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        print(f"Refit RF preferred depth in {target}", flush=True)
        return

    # `fit_session_rfs` imports `fit_rf_preferred_depth` from rf_analysis when it gets to
    # the depth fit: give it the treadmill bounds
    rf_analysis.fit_rf_preferred_depth = functools.partial(
        rf_analysis.fit_rf_preferred_depth, depth_bounds=bounds
    )

    # Fit on a scratch dataframe so that no closed-loop RF column can be overwritten
    rf_df = pd.DataFrame({"roi": np.arange(nrois)})
    rf_df = fit_session_rfs(
        rf_df,
        trials_df_all,
        session_name=session,
        flexilims_session=fs,
        filter_traces=filter_traces,
        protocol_base=PROTOCOL_BASE,
        photodiode_protocol=photodiode_protocol,
        check_nrois=True,
        use_col=use_col,
    )
    new_cols = [c for c in rf_df.columns if c != "roi"]
    assert all(c.startswith("rf_") for c in new_cols), new_cols
    assert np.array_equal(rf_df.roi.values, neurons_df.roi.values), "ROI order changed"
    neurons_df = neurons_df.copy()
    for c in new_cols:
        neurons_df[c + SUFFIX] = rf_df[c].values

    tmp = target.with_suffix(".tmp.pickle")
    neurons_df.to_pickle(tmp)
    os.replace(tmp, target)
    target.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source=str(src),
                use_col=use_col,
                protocol=PROTOCOL_BASE,
                filter_traces=filter_traces,
                columns=[c + SUFFIX for c in new_cols],
                depths=list(map(float, depth_list)),
                rf_depth_bounds=list(map(float, np.exp(bounds))),
            ),
            indent=1,
        )
    )
    print(f"Saved {target} with columns {[c + SUFFIX for c in new_cols]}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--project", required=True)
    p.add_argument("--photodiode", type=int, default=5)
    p.add_argument(
        "--annotated", action="store_true", help="use the annotated suite2p traces"
    )
    p.add_argument("--use-col", choices=["dffs", "spks"], default="dffs")
    p.add_argument(
        "--refit-depth-only",
        action="store_true",
        help="only redo the RF preferred-depth columns of an existing output file",
    )
    a = p.parse_args()
    main(a.session, a.project, a.photodiode, a.annotated, a.use_col, a.refit_depth_only)
