"""Example neurons for the reviewer grating figure: depth, optic flow and SF/TF tuning.

Port of the example-neuron figure of the thesis (`yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb`,
cell 4) to the current data: `ast_neuropil=False` traces and the cross-validated
`neurons_df_sftf.pickle`. One row per neuron:

- depth tuning (trial means and Gaussian fit) and optic flow tuning per depth, from the
  closed-loop depth protocol;
- mean response to each SF x TF for the 8 directions, around a polar plot of the maximum
  per direction (`grating_plots.plot_sftf_tuning`);
- the same for the fitted model, on the same colour scale (`grating_plots.plot_sftf_fit`).
"""

import numpy as np
import pandas as pd

from cottage_analysis.analysis import gratings, spheres
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import depth_selectivity_plots, grating_plots, rsof_plots

PROJECT = "hey2_3d-vision_foodres_20220101"
FILTER_TRACES = {"anatomical_only": 3, "ast_neuropil": False}


def load_example_session(flexilims_session, session, photodiode_protocol=5):
    """Trial tables and fits of one grating session.

    Returns:
        dict: `trials_depth` (closed-loop depth protocol, from `spheres.sync_all_recordings`),
            `trials_sftf` (grating trials with one integer column per ROI, from
            `gratings.analyze_grating_responses`), `neurons_df` and `neurons_df_sftf`.
    """
    _, trials_depth = spheres.sync_all_recordings(
        session_name=session,
        flexilims_session=flexilims_session,
        project=PROJECT,
        filter_datasets=FILTER_TRACES,
        recording_type="two_photon",
        protocol_base="SpheresPermTubeReward",
        photodiode_protocol=photodiode_protocol,
        return_volumes=True,
    )
    trials_sftf, _ = gratings.analyze_grating_responses(
        project=PROJECT,
        session=session,
        filter_datasets=FILTER_TRACES,
        photodiode_protocol=photodiode_protocol,
        protocol_base="SFTF",
    )
    folder = pipeline_utils.create_neurons_ds(
        session_name=session,
        flexilims_session=flexilims_session,
        project=PROJECT,
        conflicts="skip",
    ).path_full.parent
    return dict(
        trials_depth=trials_depth,
        trials_sftf=trials_sftf,
        neurons_df=pd.read_pickle(folder / "neurons_df.pickle"),
        neurons_df_sftf=pd.read_pickle(folder / "neurons_df_sftf.pickle"),
    )


def plot_example_neuron(
    fig,
    data,
    roi,
    top,
    fontsize_dict,
    ylim_precision_base=1,
    ylim_precision=1,
):
    """Plot one example neuron in a band of `fig` whose top is at `top` (figure fraction).

    The band is ~0.65 high: tuning curves on top, raw (left) and fitted (right) SF/TF
    responses below. Layout and settings are those of the thesis figure.
    """
    ax = fig.add_axes([0.05, top, 0.15, 0.1])
    depth_selectivity_plots.plot_depth_tuning_curve(
        neurons_df=data["neurons_df"],
        trials_df=data["trials_depth"],
        roi=roi,
        rs_thr=None,
        plot_fit=True,
        plot_smooth=False,
        linewidth=2,
        linecolor="royalblue",
        closed_loop=1,
        fontsize_dict=fontsize_dict,
        markersize=10,
        markeredgecolor="w",
        ylim_precision_base=ylim_precision_base,
        ylim_precision=ylim_precision,
        ax=ax,
    )
    ax = fig.add_axes([0.3, top, 0.15, 0.1])
    rsof_plots.plot_speed_tuning(
        trials_df=data["trials_depth"],
        roi=roi,
        is_closed_loop=1,
        nbins=10,
        which_speed="OF",
        speed_min=0.01,
        speed_max=1.5,
        speed_thr=0.01,
        of_min=1e-2,
        of_max=1e4,
        smoothing_sd=1,
        markersize=3,
        linewidth=1,
        markeredgecolor="w",
        fontsize_dict=fontsize_dict,
        legend_on=True,
        ax=ax,
    )

    trials_sftf = data["trials_sftf"]
    sf = np.sort(trials_sftf.SpatialFrequency.unique())
    tf = np.sort(trials_sftf.TemporalFrequency.unique())
    grid = dict(
        plot_grid=False,
        plot_axes=True,
        fig=fig,
        plot_y=top - 0.2,
        plot_width=0.4,
        plot_height=0.4,
    )
    vmax = grating_plots.plot_sftf_tuning(
        trials_df=trials_sftf,
        roi=roi,
        plot_x=0.05,
        fontsize_dict=fontsize_dict,
        colorbar=False,
        vmax=None,
        **grid,
    )
    grating_plots.plot_sftf_fit(
        neurons_df=data["neurons_df_sftf"],
        roi=roi,
        sf_range=[np.log(sf[0]), np.log(sf[-1])],
        tf_range=[np.log(tf[0]), np.log(tf[-1])],
        sf_ticks=sf,
        tf_ticks=tf,
        min_sigma=0.25,
        plot_x=0.55,
        fontsize_dict=fontsize_dict,
        colorbar=True,
        vmax=vmax,
        **grid,
    )
