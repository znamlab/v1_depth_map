"""Grating and depth tuning of one grating session from alternative traces, saved apart.

For the reviewer grating figure (`figures/fig_reviewer_gratings.ipynb`), whose main version
uses the production fits on `ast_neuropil=False` dF/F. This script recomputes, with the
pipeline functions and settings, but writing only to OUT_DIR/VARIANT/:

- the cross-validated grating fit, as `pipelines/depth_SFTF.py`
  (`gratings.analyze_grating_responses` then `fit_sftf_tuning(k_folds=5, fold_col="irecording")`);
- the closed-loop depth tuning, as `pipelines/analysis_pipeline.py`
  (`spheres.sync_all_recordings` then `fit_preferred_depth`, all trials and 5-fold).

VARIANT:
    dff_ast   ASt-neuropil-corrected dF/F (`ast_neuropil=True` traces, dff_ast.npy)
    spks      deconvolved spikes of the `ast_neuropil=False` traces (spks.npy)
    dff       the production traces (`ast_neuropil=False`, dff.npy), to check that the script
              reproduces the production neurons_df_sftf and neurons_df depth columns

Outputs, per session: <session>_neurons_df_sftf.pickle, <session>_trials_sftf.pickle (trial
table, ROI responses in integer columns), <session>_depth.pickle (roi + depth columns) and
<session>.json. Production files are only read.

Usage:
    python sftf_trace_variants.py SESSION VARIANT --out-dir DIR [--project P] [--n-jobs 1]
"""

import argparse
import json
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd

from cottage_analysis.analysis import find_depth_neurons, gratings, spheres
from cottage_analysis.analysis.fit_gaussian_blob import fit_sftf_tuning
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.preprocessing import synchronisation

PROJECT = "hey2_3d-vision_foodres_20220101"
PHOTODIODE_PROTOCOL = 5
VARIANTS = {
    "dff": dict(ast_neuropil=False, response="dff"),
    "dff_ast": dict(ast_neuropil=True, response="dff"),
    "spks": dict(ast_neuropil=False, response="spks"),
}
# fit_preferred_depth settings of analysis_pipeline for SpheresPermTubeReward
DEPTH_FIT = dict(depth_min=0.02, depth_max=20, niter=10, min_sigma=0.5)
DEPTH_COLUMNS = [
    "preferred_depth_closedloop",
    "depth_tuning_popt_closedloop",
    "depth_tuning_test_rsq_closedloop",
    "depth_tuning_test_spearmanr_rval_closedloop",
    "depth_tuning_test_spearmanr_pval_closedloop",
]


def load_spikes(recording_name, flexilims_session, filter_datasets=None):
    """`synchronisation.load_imaging_data` with spks.npy instead of the dF/F file."""
    ds = flz.get_datasets(
        flexilims_session=flexilims_session,
        origin_name=recording_name,
        dataset_type="suite2p_traces",
        allow_multiple=False,
        filter_datasets=filter_datasets,
    )
    nplanes = int(float(ds.extra_attributes["nplanes"]))
    spks = [np.load(ds.path_full / f"plane{i}" / "spks.npy") for i in range(nplanes)]
    return np.concatenate(spks, axis=0).T


def grating_fit(session, filter_traces, response, n_jobs):
    if response == "spks":
        synchronisation.load_imaging_data = load_spikes
    trials_df, _ = gratings.analyze_grating_responses(
        project=PROJECT,
        session=session,
        filter_datasets=filter_traces,
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        protocol_base="SFTF",
    )
    fit = fit_sftf_tuning(
        trials_df=trials_df,
        niter=10,
        min_sigma=0.25,
        k_folds=5,
        fold_col="irecording",
        n_jobs=n_jobs,
    )
    return fit, trials_df.drop(columns=["dff_stim"])


def depth_fit(session, flm, neurons_ds, filter_traces, response):
    _, trials_df = spheres.sync_all_recordings(
        session_name=session,
        flexilims_session=flm,
        project=PROJECT,
        filter_datasets=filter_traces,
        conflicts="skip",
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=PHOTODIODE_PROTOCOL,
        return_volumes=True,
        add_spikes=response == "spks",
    )
    if response == "spks":
        # fit_preferred_depth always reads dff_stim
        trials_df["dff_stim"] = trials_df["spks_stim"]
    frame_rate = flz.get_datasets(
        origin_name=session,
        dataset_type="suite2p_rois",
        project_id=PROJECT,
        flexilims_session=flm,
        return_dataseries=False,
        filter_datasets={"anatomical_only": 3},
    )[0].extra_attributes["fs"]
    # a copy of the production neurons_df, only used in memory
    neurons_df = pd.read_pickle(neurons_ds.path_full)
    for k_folds in (1, 5):
        neurons_df, _ = find_depth_neurons.fit_preferred_depth(
            trials_df=trials_df,
            neurons_df=neurons_df,
            neurons_ds=neurons_ds,
            closed_loop=1,
            choose_trials=None,
            rs_thr=None,
            rs_thr_max=None,
            still_only=False,
            still_time=0,
            frame_rate=frame_rate,
            k_folds=k_folds,
            **DEPTH_FIT,
        )
    return neurons_df[["roi"] + DEPTH_COLUMNS].copy()


def main(session, variant, out_dir, n_jobs):
    settings = VARIANTS[variant]
    filter_traces = {"anatomical_only": 3, "ast_neuropil": settings["ast_neuropil"]}
    out = Path(out_dir) / variant
    out.mkdir(parents=True, exist_ok=True)
    flm = flz.get_flexilims_session(PROJECT)
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=flm, project=PROJECT, conflicts="skip"
    )

    print(f"--- {session} {variant}: depth tuning")
    depth = depth_fit(session, flm, neurons_ds, filter_traces, settings["response"])
    depth.to_pickle(out / f"{session}_depth.pickle")

    print(f"--- {session} {variant}: grating fit")
    fit, trials = grating_fit(session, filter_traces, settings["response"], n_jobs)
    assert len(fit) == len(depth), f"{len(fit)} grating fits for {len(depth)} ROIs"
    fit.to_pickle(out / f"{session}_neurons_df_sftf.pickle")
    trials.to_pickle(out / f"{session}_trials_sftf.pickle")

    (out / f"{session}.json").write_text(
        json.dumps(
            dict(
                session=session,
                variant=variant,
                filter_traces=filter_traces,
                response=settings["response"],
                depth_fit=DEPTH_FIT,
                sftf_fit=dict(
                    niter=10, min_sigma=0.25, k_folds=5, fold_col="irecording"
                ),
            ),
            indent=1,
        )
    )
    print(f"Saved {session} {variant} to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("variant", choices=list(VARIANTS))
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-jobs", type=int, default=1, help="ROIs fitted in parallel")
    a = p.parse_args()
    main(a.session, a.variant, a.out_dir, a.n_jobs)
