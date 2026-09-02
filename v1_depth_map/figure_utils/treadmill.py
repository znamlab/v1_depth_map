import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

import flexiznam as flz
from cottage_analysis.analysis import common_utils

# Column suffix of the simulated fits run with the real trial-average plateau
# configuration (`precompute_data/fit_revision_simulation.py`), as opposed to the per-frame
# default configuration of `treadmill.simulate_and_fit_session`. It matches the `TA` suffix
# the notebooks use for the real fits, so the two families cannot be confused.
TA_SIM_SUFFIX = "_trial_average_plateau"


def plot_treadmill_protocol(
    trials_df,
    example_trials,
    fs,
    trials_df_no_cut=None,
    max_abs_rs2motor_diff_ratio=0.3,
    plot_exclude_frames=True,
    ax=None,
    save_path=None,
    add_vertical_lines=False,
    figsize=(15 / 2.54, 7 / 2.54),
    xlim=(0, 61),
    ylim=(-0.05, 0.8),
    fontsize_dict=None,
):
    """
    Plots the treadmill protocol including running speed data and stimulus
    presentation periods.

    Args:
        trials_df (pd.DataFrame): DataFrame containing 'RS_stim' and
            'max_abs_rs2motor_diff_ratio_stim'.
        example_trials (list): List of trial indices to plot.
        fs (float): Sampling frequency.
        trials_df_no_cut (pd.DataFrame, optional): DataFrame containing
            'RS_blank_pre', 'RS_stim', 'RS_blank', and 'OF_stim'.
            If None, uses trials_df. Defaults to None.
        max_abs_rs2motor_diff_ratio (float, optional): Threshold for excluding
            frames. Defaults to 0.3.
        plot_exclude_frames (bool, optional): Whether to plot the excluded
            frames. Defaults to True.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a
            new figure. Defaults to None.
        save_path (str or Path, optional): Path to save the figure as a PDF.
            Defaults to None.
        add_vertical_lines (bool, optional): Whether to add vertical lines at
            stimulus boundaries. Defaults to False.
        figsize (tuple, optional): Figure size in inches.
            Defaults to (15/2.54, 7/2.54).
        xlim (tuple, optional): X-axis limits, None to leave them automatic.
            Defaults to (0, 61).
        ylim (tuple, optional): Y-axis limits, None to leave them automatic.
            Defaults to (-0.05, 0.8).
        fontsize_dict (dict, optional): Dictionary with font size settings.
            Defaults to None.
    Returns:
        tuple: (fig, ax)
    """
    if trials_df_no_cut is None:
        trials_df_no_cut = trials_df
    if fontsize_dict is None:
        fontsize_dict = {"label": 7, "tick": 5, "legend": 5}
    # Prepare data
    data = None
    of = None
    stim_part = None

    for itrial, trial in enumerate(example_trials):
        trial_series = trials_df_no_cut.loc[trial]
        if itrial == 0:
            data = trial_series.RS_blank_pre[-20:]
            of = np.zeros_like(data) * np.nan
            stim_part = np.zeros(data.shape, dtype=int)

        assert trial_series.RS_stim.shape == trial_series.OF_stim.shape
        data = np.hstack([data, trial_series.RS_stim, trial_series.RS_blank])
        of = np.hstack(
            [
                of,
                trial_series.OF_stim,
                np.zeros_like(trial_series.RS_blank) * np.nan,
            ]
        )
        stim_part = np.hstack(
            [
                stim_part,
                np.ones(trial_series.RS_stim.shape) * trial,
                np.zeros(trial_series.RS_blank.shape),
            ]
        )

    used_data = np.zeros_like(data) * np.nan
    excluded_data = np.zeros_like(data) * np.nan

    for trial in example_trials:
        trial_series = trials_df.loc[trial]
        trial_valid = trial_series.RS_stim
        ok_mask = (
            trial_series.max_abs_rs2motor_diff_ratio_stim < max_abs_rs2motor_diff_ratio
        )
        trial_used = np.where(ok_mask, trial_valid, np.nan)
        trial_excluded = np.where(ok_mask, np.nan, trial_valid)

        trial_indices = np.where(stim_part == trial)[0]
        if len(trial_indices) > 0:
            end_ind = trial_indices[-1]
            excluded_data[end_ind - len(trial_excluded) + 1 : end_ind + 1] = (
                trial_excluded
            )
            used_data[end_ind - len(trial_used) + 1 : end_ind + 1] = trial_used

    # Plotting
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    time_axis = np.arange(len(data)) / fs
    if plot_exclude_frames:
        ax.plot(time_axis, data, color="k", lw=1, clip_on=False, label="All Frames")
        ax.plot(time_axis, excluded_data, color="grey", lw=2, label="Excluded Frames")
    else:
        ax.plot(time_axis, data, color="k", lw=1, clip_on=False)
    ax.scatter(
        time_axis,
        used_data,
        s=5,
        color="dodgerblue",
        zorder=20,
        # lw=2,
        label="Analysed Frames",
    )

    has_of = ~np.isnan(of)
    ax.fill_between(
        time_axis,
        0.9,
        1,
        where=has_of,
        color="grey",
        alpha=0.3,
        transform=ax.get_xaxis_transform(),
        zorder=-1,
        label="Sphere Presentation",
    )

    edges = np.where(np.diff(np.concatenate(([False], has_of, [False]))))[0].reshape(
        -1, 2
    )
    if add_vertical_lines:
        for start, end in edges:
            ax.axvline(start / fs, color="black", linestyle="--", alpha=0.5)
            ax.axvline(end / fs, color="black", linestyle="--", alpha=0.5)

    # Scale bar
    x0, y0 = 1, 0.2
    x_len, y_len = 2, 0.1
    ax.plot(
        [x0, x0, x0 + x_len],
        [y0 + y_len, y0, y0],
        color="k",
        lw=2,
        clip_on=False,
        solid_joinstyle="miter",
    )
    ax.text(
        x0 + x_len / 2,
        y0 - (y_len * 0.15),
        f"{int(x_len)} s",
        ha="center",
        va="top",
        fontsize=fontsize_dict["legend"],
        clip_on=False,
    )
    ax.text(
        x0 - (x_len * 0.1),
        y0 + y_len / 2,
        f"{int(y_len * 100)} cm/s",
        ha="right",
        va="center",
        rotation=90,
        fontsize=fontsize_dict["legend"],
        clip_on=False,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Random dots for spheres
    for start, end in edges:
        x_start = start / fs + 1
        x_end = end / fs - 1
        n_dots = np.random.randint(4, 8)
        rand_x = np.linspace(x_start, x_end, n_dots) + np.random.uniform(
            -0.5, 0.5, n_dots
        )
        rand_y = np.random.uniform(0.92, 0.98, n_dots)
        ax.scatter(
            rand_x,
            rand_y,
            color="black",
            s=10,
            zorder=0,
            transform=ax.get_xaxis_transform(),
        )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=fontsize_dict["legend"],
        bbox_to_anchor=(0.0, 0.9),
    )

    _save_pdf(fig, save_path)

    return fig, ax


def _save_pdf(fig, save_path):
    """Save a figure as a pdf with editable fonts, if save_path is not None."""
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    old_pdf_fonttype = mpl.rcParams["pdf.fonttype"]
    old_ps_fonttype = mpl.rcParams["ps.fonttype"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    fig.savefig(save_path, format="pdf", bbox_inches="tight", transparent=True)
    mpl.rcParams["pdf.fonttype"] = old_pdf_fonttype
    mpl.rcParams["ps.fonttype"] = old_ps_fonttype


def plot_treadmill_stim_sampling(
    trials_df,
    ax=None,
    max_abs_rs2motor_diff_ratio=0.3,
    fontsize_dict={"label": 14, "tick": 12, "legend": 12},
    markersize=5,
    trial_markersize=20,
    legend_on=True,
    legend_kwargs=None,
    ylim=None,
    figsize=(7 / 2.54, 7 / 2.54),
    save_path=None,
):
    """Plot how the treadmill stimulus samples the running speed / optic flow plane.

    Single imaging frames are shown in the background, with the per-trial averages on
    top. Both axes are logarithmic (base 2) and ticked at the motor speeds and expected
    optic flow speeds of the protocol.

    Args:
        trials_df (pd.DataFrame): DataFrame containing 'RS_stim', 'OF_stim',
            'MotorSpeed_stim', 'expected_optic_flow_stim' and, optionally,
            'max_abs_rs2motor_diff_ratio_stim'.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. If None, creates a new
            figure. Defaults to None.
        max_abs_rs2motor_diff_ratio (float, optional): Threshold for excluding frames
            where the running speed does not match the motor speed. Set to None to keep
            all frames. Defaults to 0.3.
        fontsize_dict (dict, optional): Dictionary of fontsizes.
        markersize (float, optional): Marker size for single frames. Defaults to 5.
        trial_markersize (float, optional): Marker size for trial averages.
            Defaults to 20.
        legend_on (bool, optional): Whether to add the legend. Defaults to True.
        legend_kwargs (dict, optional): Overrides for the legend keyword arguments.
            Defaults to None.
        ylim (tuple, optional): (ymin, ymax) for the optic flow axis, in degrees/s.
            Use this to clip rare single-frame glitches (e.g. a one-frame near-stall
            in the rendered eye position, giving a spuriously tiny optic flow value)
            without excluding them from the underlying data. Defaults to None, i.e.
            matplotlib's automatic limits.
        figsize (tuple, optional): Figure size in inches, ignored if ax is given.
            Defaults to (7/2.54, 7/2.54).
        save_path (str or Path, optional): Path to save the figure as a PDF.
            Defaults to None.

    Returns:
        tuple: (fig, ax)
    """
    if (max_abs_rs2motor_diff_ratio is not None) and (
        "max_abs_rs2motor_diff_ratio_stim" in trials_df.columns
    ):
        trials_df = common_utils.filter_trials_by_rs2motor(
            trials_df,
            max_rs2motor_diff=max_abs_rs2motor_diff_ratio,
            col2filter=["RS_stim", "OF_stim"],
        )

    # Per-frame running speed (cm/s) and optic flow (degrees/s). Only strictly positive
    # values can be shown on log axes.
    rs_frames = np.hstack(trials_df.RS_stim.values) * 100
    of_frames = np.degrees(np.hstack(trials_df.OF_stim.values))
    ok = (rs_frames > 0) & (of_frames > 0)
    rs_frames, of_frames = rs_frames[ok], of_frames[ok]

    # Per-trial averages, in the same units. Trials without any valid frame are NaN and
    # dropped by the positivity check below.
    def trial_mean(x):
        return np.nanmean(x) if np.any(~np.isnan(x)) else np.nan

    rs_trials = trials_df.RS_stim.map(trial_mean).values * 100
    of_trials = np.degrees(trials_df.OF_stim.map(trial_mean).values)
    ok = (rs_trials > 0) & (of_trials > 0)
    rs_trials, of_trials = rs_trials[ok], of_trials[ok]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(1, 1, 1)
    else:
        fig = ax.get_figure()

    ax.scatter(
        rs_frames,
        of_frames,
        alpha=0.1,
        color="k",
        s=markersize,
        edgecolors="none",
        label="Single frames",
        rasterized=True,
    )
    ax.scatter(
        rs_trials,
        of_trials,
        alpha=0.8,
        color="dodgerblue",
        s=trial_markersize,
        edgecolors="white",
        linewidths=0.1,
        label="Trial averages",
    )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    if ylim is not None:
        ax.set_ylim(*ylim)

    motor_speeds = np.unique(
        trials_df.MotorSpeed_stim.map(np.nanmedian).dropna().round()
    )
    of_speeds = np.unique(
        trials_df.expected_optic_flow_stim.map(np.nanmedian).dropna().round()
    )
    rs_ticks = motor_speeds[motor_speeds > 0].astype(int)
    of_ticks = of_speeds[of_speeds > 0].astype(int)

    ax.set_xticks(rs_ticks)
    ax.set_xticklabels([f"{x}" for x in rs_ticks], fontsize=fontsize_dict["tick"])
    ax.set_yticks(of_ticks)
    ax.set_yticklabels([f"{x}" for x in of_ticks], fontsize=fontsize_dict["tick"])
    ax.set_xlabel("Running speed (cm/s)", fontsize=fontsize_dict["label"])
    ax.set_ylabel("Optic flow speed (°/s)", fontsize=fontsize_dict["label"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(min(rs_ticks), max(rs_ticks))
    ax.spines["left"].set_bounds(min(of_ticks), max(of_ticks))

    if legend_on:
        kwargs = dict(
            loc="upper right",
            bbox_to_anchor=(1, 1.3),
            fontsize=fontsize_dict["legend"],
            frameon=False,
        )
        kwargs.update(legend_kwargs or {})
        ax.legend(**kwargs)

    _save_pdf(fig, save_path)

    return fig, ax


def load_treadmill_population_neurons_df(
    flm_sess_rev,
    protocol_base="SpheresTubeMotor",
    tdecay=2,
    trise=0.15,
    percentile=95,
    load_simulated=False,
    load_simulated_trial_average=False,
):
    """Load and filter the population `neurons_df` for treadmill-protocol sessions.

    Finds every session in the project that ran the treadmill protocol, loads and
    concatenates their `neurons_df` (and, if `load_simulated`, the matching
    simulated-response dataframes), flags depth- and RS/OF-tuned neurons, and adds
    ellipse/Gaussian-fit-derived columns for both the closed-loop and treadmill fits.

    Significance of the RS/OF (`g2d`) fit is assessed against an empirical null:
    the negative tail of the R-squared distribution is used to estimate the null's
    width, and neurons above `percentile` of that null are flagged as tuned
    (`rsof_neuron`/`rsof_neuron_treadmill`), see
    `cottage_analysis.analysis.common_utils.empirical_null_threshold`.

    Args:
        flm_sess_rev: Flexilims session for the project holding the treadmill data.
        protocol_base (str, optional): Protocol name used to find treadmill sessions.
            Defaults to "SpheresTubeMotor".
        tdecay (float, optional): Calcium decay time constant used to pick the
            simulated-response file to load. Defaults to 2.
        trise (float, optional): Calcium rise time constant used to pick the
            simulated-response file to load. Defaults to 0.15.
        percentile (float, optional): Percentile of the empirical null distribution
            used as the significance threshold for `rsof_neuron`/
            `rsof_neuron_treadmill`. Defaults to 95.
        load_simulated (bool, optional): Whether to load and process the
            simulated-response parquet files (`simul_df_treadmill`/`simul_df_spheres`).
            These are only needed for simulated-vs-real comparisons and are slower to
            load, so this defaults to False and can be set to True to also load them.
        load_simulated_trial_average (bool, optional): Whether to also load the fits of the
            same simulated treadmill responses run with the real trial-average plateau
            configuration (`precompute_data/fit_revision_simulation.py`), rather than the
            per-frame configuration `simulate_and_fit_session` used. They are merged into
            `simul_df_treadmill` with the `TA_SIM_SUFFIX` suffix, alongside derived
            ellipse geometry, so the two fit families sit side by side on the same row.
            Requires `load_simulated`. Defaults to False.

    Returns:
        tuple: (neurons_df, simul_df_treadmill, simul_df_spheres, valid_sessions,
            treadmill_sessions). `simul_df_treadmill`/`simul_df_spheres` are None if
            `load_simulated` is False.
    """
    from cottage_analysis.io_module import suite2p as s2p_io
    from cottage_analysis.analysis import fit_gaussian_blob as fit_gb

    mice = flz.get_entities(datatype="mouse", flexilims_session=flm_sess_rev)
    all_sessions = flz.get_entities(datatype="session", flexilims_session=flm_sess_rev)

    treadmill_sessions = {}
    for _, mouse_data in mice.iterrows():
        sessions = all_sessions[all_sessions.origin_id == mouse_data.id]
        for session_name, sess_data in sessions.iterrows():
            recordings = flz.get_children(
                parent_id=sess_data.id,
                flexilims_session=flm_sess_rev,
                children_datatype="recording",
            )
            if not len(recordings):
                continue
            if protocol_base in recordings.protocol.values:
                treadmill_sessions[session_name] = recordings

    if load_simulated_trial_average and not load_simulated:
        raise ValueError(
            "load_simulated_trial_average=True needs load_simulated=True: the "
            "trial-average fits are merged into simul_df_treadmill."
        )

    valid_sessions = []
    all_dfs = []
    simulated_responses = dict(treadmill=[], spheres=[], treadmill_trial_average=[])
    for session_name in treadmill_sessions:
        neurons_ds = flz.get_datasets(
            origin_name=session_name,
            dataset_type="neurons_df",
            flexilims_session=flm_sess_rev,
            allow_multiple=False,
        )
        if neurons_ds is None:
            print(f"Neurons dataset not found for session {session_name}")
            continue
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        suite2p_ds = flz.get_datasets(
            origin_name=session_name,
            dataset_type="suite2p_rois",
            filter_datasets={"annotated": True},
            flexilims_session=flm_sess_rev,
            allow_multiple=False,
        )
        neurons_df["is_cell"] = s2p_io.load_is_cell(suite2p_ds.path_full)
        if "is_depth_neuron" not in neurons_df.columns:
            print(
                f"Depth selectivity not computed for session {session_name}, run basic analysis first"
            )
            continue
        elif "is_depth_neuron_treadmill" not in neurons_df.columns:
            print(
                f"Treadmill data not processed for session {session_name}, run treadmill analysis first"
            )
            continue
        valid_sessions.append(session_name)
        neurons_df["session"] = session_name
        neurons_df["roi_uid"] = session_name + "_" + neurons_df.roi.astype(str)
        all_dfs.append(neurons_df)
        if load_simulated:
            for which in ["treadmill", "spheres"]:
                simul_path = neurons_ds.path_full.with_name(
                    f"simulated_responses_fit_{which}_{tdecay}_{trise}_circular.parquet"
                )
                if simul_path.exists():
                    simul_df = pd.read_parquet(simul_path)
                    simul_df["session"] = session_name
                    simul_df["roi_uid"] = session_name + "_" + simul_df.roi.astype(str)
                    simulated_responses[which].append(simul_df)
                else:
                    print(
                        f"Simulated responses not found for {which} for session {session_name}"
                    )
        if load_simulated_trial_average:
            # Fits of the SAME simulated dF/F, but with the real trial-average plateau
            # config. Written by `precompute_data/fit_revision_simulation.py`; `fake_dff`
            # is not repeated in this file, it stays in the one loaded above.
            ta_path = neurons_ds.path_full.with_name(
                "simulated_responses_fit_treadmill_trial_average_plateau"
                f"_{tdecay}_{trise}_circular.parquet"
            )
            if ta_path.exists():
                ta_df = pd.read_parquet(ta_path)
                ta_df["roi_uid"] = session_name + "_" + ta_df.roi.astype(str)
                simulated_responses["treadmill_trial_average"].append(ta_df)
            else:
                print(
                    "Trial-average simulated fits not found for session "
                    f"{session_name}: run precompute_data/fit_revision_simulation.py"
                )
    print(
        f"{len(valid_sessions)}/{len(treadmill_sessions)} valid sessions with treadmill depth data"
    )

    neurons_df = pd.concat(all_dfs, ignore_index=True)
    if load_simulated:
        simul_df_treadmill = pd.concat(
            simulated_responses["treadmill"], ignore_index=True
        )
        simul_df_spheres = pd.concat(simulated_responses["spheres"], ignore_index=True)
        if load_simulated_trial_average:
            ta_frames = simulated_responses["treadmill_trial_average"]
            if not ta_frames:
                raise FileNotFoundError(
                    "load_simulated_trial_average=True but no session had a "
                    "simulated_responses_fit_treadmill_trial_average_plateau parquet."
                )
            ta_all = pd.concat(ta_frames, ignore_index=True).drop(columns=["roi"])
            ta_all = ta_all.rename(
                columns={
                    col: f"{col}{TA_SIM_SUFFIX}"
                    for col in ta_all.columns
                    if col != "roi_uid"
                }
            )
            simul_df_treadmill = simul_df_treadmill.merge(
                ta_all, on="roi_uid", how="left"
            )
    else:
        simul_df_treadmill = None
        simul_df_spheres = None

    # Find tuned cells
    neurons_df = neurons_df[neurons_df["is_cell"]].copy()

    common_utils.add_one_sided_spearman_significance(
        neurons_df,
        rval_col="depth_tuning_test_spearmanr_rval_closedloop",
        pval_col="depth_tuning_test_spearmanr_pval_closedloop",
        out_col="is_depth_neuron",
    )
    common_utils.add_one_sided_spearman_significance(
        neurons_df,
        rval_col="depth_tuning_test_spearmanr_rval_closedloop_treadmill",
        pval_col="depth_tuning_test_spearmanr_pval_closedloop_treadmill",
        out_col="is_depth_neuron_treadmill",
    )

    # Empirical-null significance test for the RS/OF (g2d) fit, closed-loop and treadmill
    for which, flag_col in [
        ("", "rsof_neuron"),
        ("_treadmill", "rsof_neuron_treadmill"),
    ]:
        rsq_col = f"rsof_test_rsq_closedloop_g2d{which}"
        rsq_vals = pd.to_numeric(neurons_df[rsq_col], errors="coerce").dropna().values
        finite_vals = rsq_vals[np.isfinite(rsq_vals) & (rsq_vals >= -1)]
        thr, _ = common_utils.empirical_null_threshold(
            finite_vals, percentile=percentile
        )
        neurons_df[flag_col] = pd.to_numeric(neurons_df[rsq_col], errors="coerce") > thr

    print(
        f"{neurons_df.is_depth_neuron.sum()}/{len(neurons_df)} depth tuned neurons in closed loop"
    )
    print(
        f"{neurons_df.is_depth_neuron_treadmill.sum()}/{len(neurons_df)} depth tuned neurons in closed loop treadmill"
    )
    print(
        f"{neurons_df.rsof_neuron_treadmill.sum()}/{len(neurons_df)} rsof neurons in closed loop treadmill"
    )

    # Add ellipse properties calculated from fit parameters, for both real and simulated data
    for which in ["_treadmill", ""]:
        popt_col = neurons_df[f"rsof_popt_closedloop_g2d{which}"]
        neurons_df[f"g2d_theta{which}"] = popt_col.apply(fit_gb.get_gaussian_angle)
        neurons_df[f"g2d_semimajor{which}"] = popt_col.apply(
            fit_gb.get_semimajor_length
        )
        neurons_df[f"g2d_semiminor{which}"] = popt_col.apply(
            fit_gb.get_semiminor_length
        )
        neurons_df[f"g2d_eccentricity{which}"] = popt_col.apply(
            fit_gb.get_gaussian_eccentricity
        )
        neurons_df[f"g2d_preferred_RS{which}"] = popt_col.apply(fit_gb.get_preferred_rs)
        neurons_df[f"g2d_preferred_OF{which}"] = popt_col.apply(fit_gb.get_preferred_of)

    if load_simulated:
        simul_df_treadmill["g2d_theta_treadmill"] = simul_df_treadmill[
            "popt_simulated"
        ].apply(fit_gb.get_gaussian_angle)
        simul_df_treadmill["g2d_eccentricity_treadmill"] = simul_df_treadmill[
            "popt_simulated"
        ].apply(fit_gb.get_gaussian_eccentricity)

        simul_df_spheres["g2d_theta"] = simul_df_spheres["popt_simulated"].apply(
            fit_gb.get_gaussian_angle
        )
        simul_df_spheres["g2d_eccentricity"] = simul_df_spheres["popt_simulated"].apply(
            fit_gb.get_gaussian_eccentricity
        )

        if load_simulated_trial_average:
            # `min_sigma` is recorded by the fit, so read it rather than relying on the
            # helpers' default (as `add_trial_average_rsof_columns` does for the real
            # fits). Semi-axes are added here too, which the per-frame simulated family
            # lacks - that is why the notebooks recompute elongation inline for it.
            min_sigma_sim = float(
                pd.to_numeric(
                    simul_df_treadmill[f"min_sigma{TA_SIM_SUFFIX}"], errors="coerce"
                )
                .dropna()
                .iloc[0]
            )
            popt_ta = simul_df_treadmill[f"popt_simulated{TA_SIM_SUFFIX}"]
            ok = popt_ta.apply(
                lambda x: isinstance(x, (list, np.ndarray)) and len(x) >= 6
            )
            simul_df_treadmill.loc[ok, f"g2d_theta{TA_SIM_SUFFIX}"] = popt_ta[ok].apply(
                fit_gb.get_gaussian_angle
            )
            for name, fn in [
                (f"g2d_semimajor{TA_SIM_SUFFIX}", fit_gb.get_semimajor_length),
                (f"g2d_semiminor{TA_SIM_SUFFIX}", fit_gb.get_semiminor_length),
                (f"g2d_eccentricity{TA_SIM_SUFFIX}", fit_gb.get_gaussian_eccentricity),
            ]:
                simul_df_treadmill.loc[ok, name] = popt_ta[ok].apply(
                    lambda x, fn=fn: fn(x, min_sigma=min_sigma_sim)
                )
            print(
                f"{int(ok.sum())}/{len(simul_df_treadmill)} simulated ROIs with a "
                "trial-average plateau fit"
            )

    return (
        neurons_df,
        simul_df_treadmill,
        simul_df_spheres,
        valid_sessions,
        treadmill_sessions,
    )


def add_trial_average_rsof_columns(
    neurons_df,
    ta_suffix="_treadmill_trial_average_plateau",
    percentile=95,
    models=("gof", "grs", "gadd", "g2d", "gratio"),
    null_method="empirical",
    verbose=True,
):
    """Derive ellipse geometry and significance flags from the trial-averaged g2d fits.

    `load_treadmill_population_neurons_df` only does this for the per-frame fits (the ""
    and "_treadmill" suffixes). The trial-averaged fits - one sample per trial rather than
    per imaging frame - live in separate columns written by
    `precompute_data/fit_revision_treadmill.py` and merged into each session's
    `neurons_df.pickle`, so they are already loaded but have no derived columns.

    Adds, all suffixed with `ta_suffix`: `g2d_theta` (degrees, wrapped to [-45, 135]),
    `g2d_eccentricity`, `g2d_semimajor`, `g2d_semiminor`, `g2d_preferred_RS` (cm/s),
    `g2d_preferred_OF` (deg/s) and `best_model`; plus one `<rsq_col>_sig` flag per model.

    `g2d_eccentricity` is the standard geometric eccentricity of the tuning ellipse,
    `sqrt(1 - sigma_minor^2 / sigma_major^2)`.

    Fits whose Gaussian is a "ridge" spanning the whole sampled plane are kept as they
    are: `exp(log_sigma2)` overflows on the unbounded axis, so they simply come out at
    eccentricity 1 with a well-defined angle.

    Args:
        neurons_df (pd.DataFrame): Population dataframe, modified in place.
        ta_suffix (str, optional): Column suffix of the trial-averaged fits.
            "_treadmill_trial_average" is the default ("model") onset detection,
            "_treadmill_trial_average_plateau" the "plateau" one. Defaults to the latter,
            matching `tread_kwargs=dict(method="plateau")`.
        percentile (float, optional): Percentile of the empirical null used as the
            significance threshold. Defaults to 95.
        models (tuple, optional): Models refit on the trial averages, used for the
            significance flags and `best_model`.
        null_method (str, optional): Passed to `common_utils.empirical_null_threshold`.
            Defaults to "empirical" (percentile of the mirrored negative tail itself),
            matching the default of `common_utils.add_rsq_significance`. "gaussian" fits
            a zero-mean Gaussian to that tail instead.
        verbose (bool, optional): Whether to print per-model thresholds and counts.
            Defaults to True.

    Returns:
        float: `min_sigma` recorded by the fit, needed by `plot_RS_OF_fit` and the
            semi-axis helpers.
    """
    from cottage_analysis.analysis import fit_gaussian_blob as fit_gb

    ta = ta_suffix
    popt_col = f"rsof_popt_closedloop_g2d{ta}"
    rsq_cols = {m: f"rsof_test_rsq_closedloop_{m}{ta}" for m in models}

    missing = [c for c in [popt_col, *rsq_cols.values()] if c not in neurons_df.columns]
    if missing:
        raise KeyError(
            f"Missing trial-average columns: {missing}\n"
            "Run precompute_data/fit_revision_treadmill.py (--only treadmill) and "
            "then --merge."
        )

    # min_sigma is recorded by the fit itself - read it rather than assuming 0.25. The k1
    # and k5 pickles both carry this column, so merge_fit_dataframes disambiguates them
    # as _x/_y (same pre-existing artefact as the sphere and per-frame treadmill
    # families); either is fine.
    ms_col = next(
        (
            c
            for c in (
                f"rsof_minSigma_closedloop_g2d{ta}",
                f"rsof_minSigma_closedloop_g2d{ta}_x",
            )
            if c in neurons_df.columns
        ),
        None,
    )
    if ms_col is None:
        raise KeyError(f"No rsof_minSigma_closedloop_g2d{ta}[_x] column")
    min_sigma = float(
        pd.to_numeric(neurons_df[ms_col], errors="coerce").dropna().unique()[0]
    )

    # ---- Ellipse geometry -------------------------------------------------------------
    ok = neurons_df[popt_col].apply(
        lambda x: isinstance(x, (list, np.ndarray)) and len(x) >= 6
    )
    # Legacy parameterisation: get_gaussian_angle takes no min_sigma, the others do
    neurons_df.loc[ok, f"g2d_theta{ta}"] = neurons_df.loc[ok, popt_col].apply(
        fit_gb.get_gaussian_angle
    )
    for name, fn in [
        (f"g2d_eccentricity{ta}", fit_gb.get_gaussian_eccentricity),
        (f"g2d_semimajor{ta}", fit_gb.get_semimajor_length),
        (f"g2d_semiminor{ta}", fit_gb.get_semiminor_length),
    ]:
        neurons_df.loc[ok, name] = neurons_df.loc[ok, popt_col].apply(
            lambda x, fn=fn: fn(x, min_sigma=min_sigma)
        )
    neurons_df.loc[ok, f"g2d_preferred_RS{ta}"] = neurons_df.loc[ok, popt_col].apply(
        fit_gb.get_preferred_rs
    )
    neurons_df.loc[ok, f"g2d_preferred_OF{ta}"] = neurons_df.loc[ok, popt_col].apply(
        fit_gb.get_preferred_of
    )

    # ---- Significance from an empirical null ------------------------------------------
    for model, col in rsq_cols.items():
        vals = pd.to_numeric(neurons_df[col], errors="coerce").dropna().values
        vals = vals[np.isfinite(vals) & (vals >= -1)]  # drop sentinel values
        thr, sigma = common_utils.empirical_null_threshold(
            vals, percentile=percentile, method=null_method
        )
        neurons_df[f"{col}_sig"] = pd.to_numeric(neurons_df[col], errors="coerce") > thr
        if verbose:
            print(
                f"{model:8s} sigma={sigma:.4f}  threshold={thr:.4f}  "
                f"passing={neurons_df[f'{col}_sig'].mean() * 100:5.1f}%"
            )

    # Best model per neuron (highest test R-squared), as a model name not a column name
    rsq_frame = neurons_df[list(rsq_cols.values())].apply(
        pd.to_numeric, errors="coerce"
    )
    neurons_df[f"best_model{ta}"] = rsq_frame.idxmax(axis=1).map(
        {v: k for k, v in rsq_cols.items()}
    )

    if verbose:
        print(f"popt_col = {popt_col}\n  -> min_sigma = {min_sigma}")
        print(f"ellipse properties for {int(ok.sum())} neurons")

    return min_sigma


def compute_treadmill_rsof_bins(trials_df_tm):
    """Compute RS/OF bin edges and tick_dict matching the discrete motor
    speeds/optic flows used on the treadmill.

    Args:
        trials_df_tm (pd.DataFrame): treadmill trials_df with 'MotorSpeed_stim'
            and 'expected_optic_flow_stim' columns.

    Returns:
        tuple: (rs_bins, of_bins, tick_dict)
    """
    motor_speeds = np.round(np.unique(trials_df_tm.MotorSpeed_stim.map(np.nanmedian)))
    ms_log = np.log2(motor_speeds)
    rs_bw = np.median(np.diff(ms_log))
    # one bin per motor speed, plus one extra below the slowest one to catch trials
    # where the mouse ran slower than the wheel. Nothing above the fastest motor
    # speed: that bin can only ever be empty.
    rs_bins = 2 ** (ms_log[0] - rs_bw * 1.5 + rs_bw * np.arange(len(motor_speeds) + 2))
    rs_bins = np.insert(rs_bins, 0, 0)

    of_speeds = np.round(
        np.unique(trials_df_tm.expected_optic_flow_stim.map(np.nanmedian))
    )
    of_log = np.log2(of_speeds)
    of_bw = np.median(np.diff(of_log))
    of_bins = 2 ** np.arange(of_log[0] - of_bw * 0.5, of_log[-1] + of_bw, of_bw)
    of_bins = np.insert(of_bins, 0, 0)

    rs_logbin = np.log2(rs_bins[1:])
    rs_bin_middle = np.diff(rs_logbin) / 2 + rs_logbin[:-1]
    of_logbin = np.log2(of_bins[1:])
    of_bin_middle = np.diff(of_logbin) / 2 + of_logbin[:-1]
    tick_dict = dict(
        rs_tick_select=rs_bin_middle,
        rs_tick_values=(2**rs_bin_middle).astype(int),
        of_tick_select=of_bin_middle,
        of_tick_values=(2**of_bin_middle).astype(int),
    )
    return rs_bins, of_bins, tick_dict
