"""RF fits of one session on the suite2p spikes, saved next to neurons_df.

Runs the RF step of the analysis pipeline (`analysis_pipeline.fit_session_rfs`, same
stimulus, lag, hyperparameter grid, ipsi fit and RF depth fit) with the deconvolved
spikes instead of dF/F, for each sphere protocol of the session. The other columns are
copied from `neurons_df.pickle`, which is never modified; the result is saved as
`neurons_df_spks.pickle` in the same folder (same column names), so analyses can load
either file.

Usage:
    python rf_fit_spikes.py SESSION --project PROJECT [--protocols A,B] [--photodiode 5] [--annotated]
"""

import argparse
import json
import os

import pandas as pd
import flexiznam as flz

from cottage_analysis.analysis import spheres
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.pipelines.analysis_pipeline import fit_session_rfs


def main(session, project, protocols, photodiode_protocol, annotated, use_col="spks"):
    fs = flz.get_flexilims_session(project)
    neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session, flexilims_session=fs, project=project, conflicts="skip"
    )
    src = neurons_ds.path_full
    target = src.with_name(f"neurons_df_{use_col}.pickle")
    neurons_df = pd.read_pickle(src)
    print(f"{session}: {len(neurons_df)} ROIs from {src}")
    filter_traces = {"anatomical_only": 3, "ast_neuropil": False}
    if annotated:
        filter_traces["annotated"] = True
    for protocol_base in protocols:
        print(f"--- {protocol_base}")
        _, trials_df_all = spheres.sync_all_recordings(
            session_name=session,
            flexilims_session=fs,
            project=project,
            filter_datasets=filter_traces,
            conflicts="skip",
            recording_type="two_photon",
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            return_volumes=True,
        )
        neurons_df = fit_session_rfs(
            neurons_df,
            trials_df_all,
            session_name=session,
            flexilims_session=fs,
            filter_traces=filter_traces,
            protocol_base=protocol_base,
            photodiode_protocol=photodiode_protocol,
            check_nrois=True,
            use_col=use_col,
        )
    tmp = target.with_suffix(".tmp.pickle")
    neurons_df.to_pickle(tmp)
    os.replace(tmp, target)
    target.with_suffix(".json").write_text(
        json.dumps(
            dict(
                source=str(src),
                use_col=use_col,
                protocols=protocols,
                filter_traces=filter_traces,
            ),
            indent=1,
        )
    )
    print(f"Saved {target}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("session")
    p.add_argument("--project", required=True)
    p.add_argument(
        "--protocols",
        default="SpheresPermTubeReward",
        help="comma-separated sphere protocol bases fitted in this session",
    )
    p.add_argument("--photodiode", type=int, default=5)
    p.add_argument(
        "--annotated", action="store_true", help="use the annotated suite2p traces"
    )
    p.add_argument(
        "--use-col",
        choices=["spks", "dffs"],
        default="spks",
        help="dffs only to check that the stored dF/F fits are reproduced (writes neurons_df_dffs.pickle)",
    )
    a = p.parse_args()
    main(
        a.session,
        a.project,
        a.protocols.split(","),
        a.photodiode,
        a.annotated,
        a.use_col,
    )
