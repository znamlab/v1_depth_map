import pickle
from pathlib import Path
import scipy
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import flexiznam as flz
from cottage_analysis.depth_analysis.plotting.plotting_utils import (
    calculate_R_squared,
    get_PSTH,
)
from cottage_analysis.depth_analysis.depth_preprocess import process_params


def get_or_load_fit(
    recording,
    seed,
    prop_fit,
    stim_dict_fit,
    target_folder,
    speed_thr_cal,
    min_sigma,
    data,
    redo=False,
):

    fname = f"{'_'.join(recording.genealogy)}_gaussian_fit_{seed}_{prop_fit}.csv"
    target = Path(target_folder / fname)
    if target.exists() and not redo:
        return pd.read_csv(target)

    # exclude non-cells
    # The ROI no. for all cells (excluding non-cells
    iscell = data["iscell"]
    dffs_ast = data["dffs_ast"]
    which_rois = (np.arange(dffs_ast.shape[0]))[iscell.astype("bool")]

    img_VS_all = data["param_logger"]
    ops = data["ops"]
    frame_rate = int(ops["fs"])
    depth_list = img_VS_all["Depth"].unique()
    depth_list = np.round(depth_list, 2)
    depth_list = depth_list[~np.isnan(depth_list)].tolist()
    depth_list.remove(-99.99)
    depth_list.sort()
    depth_list[-1] = 6  # dirty mix up between 6.0 and 6

    # find max depth to initialise fit
    max_depths = get_max_depth(which_rois, dffs_ast, depth_list, stim_dict_fit)
    # now perform the fit with fit_trials
    print("MIN SIGMA", str(min_sigma), flush=True)
    batch_num = 5
    depth_min = 2
    depth_max = 2000
    speeds = img_VS_all.MouseZ.diff() / img_VS_all.HarpTime.diff()  # m/s
    speeds[0] = 0
    speed_arr_fit, _ = process_params.create_speed_arr(
        speeds=speeds,
        depth_list=depth_list,
        stim_dict=stim_dict_fit,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
    gaussian_depth_fit_df = gauss_fit(
        which_rois,
        dffs_ast,
        depth_list,
        stim_dict_fit,
        frame_rate,
        speed_arr_fit,
        max_depths,
        speed_thr_cal,
        batch_num,
        depth_min,
        depth_max,
        min_sigma=min_sigma,
    )
    return (gaussian_depth_fit_df,)


def get_or_load_anova(
    recording, target_folder, seed, prop_fit, data, stim_dict, speed_thr_cal, redo=False
):
    fname = f"{'_'.join(recording.genealogy)}_anovas_{seed}_{prop_fit}.npy"
    target = Path(target_folder / fname)
    if target.exists() and not redo:
        return np.load(target)

    iscell = data["iscell"]
    dffs_ast = data["dffs_ast"]
    which_rois = (np.arange(dffs_ast.shape[0]))[iscell.astype("bool")]
    img_VS_all = data["param_logger"]
    ops = data["ops"]
    frame_rate = int(ops["fs"])
    depth_list = img_VS_all["Depth"].unique()
    depth_list = np.round(depth_list, 2)
    depth_list = depth_list[~np.isnan(depth_list)].tolist()
    depth_list.remove(-99.99)
    depth_list.sort()
    depth_list[-1] = 6  # dirty mix up between 6.0 and 6

    speeds = img_VS_all.MouseZ.diff() / img_VS_all.HarpTime.diff()  # m/s
    speeds[0] = 0
    speed_arr, _ = process_params.create_speed_arr(
        speeds=speeds,
        depth_list=depth_list,
        stim_dict=stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )

    anova = get_anova(
        which_rois,
        dffs_ast,
        depth_list,
        stim_dict,
        frame_rate,
        speed_arr,
        speed_thr_cal=speed_thr_cal,
    )
    np.save(target, anova)
    return anova


def get_or_load_psth(
    recording, target_folder, seed, prop_fit, data, stim_dict_raster, redo=False
):
    # compute psth with other stim dict
    fname = f"{'_'.join(recording.genealogy)}_psth_{seed}_{prop_fit}.npy"
    target = Path(target_folder / fname)
    if target.exists() and not redo:
        return np.load(target)

    # exclude non-cells
    # The ROI no. for all cells (excluding non-cells
    iscell = data["iscell"]
    dffs_ast = data["dffs_ast"]
    which_rois = (np.arange(dffs_ast.shape[0]))[iscell.astype("bool")]
    img_VS_all = data["param_logger"]
    ops = data["ops"]
    frame_rate = int(ops["fs"])
    depth_list = img_VS_all["Depth"].unique()
    depth_list = np.round(depth_list, 2)
    depth_list = depth_list[~np.isnan(depth_list)].tolist()
    depth_list.remove(-99.99)
    depth_list.sort()
    depth_list[-1] = 6  # dirty mix up between 6.0 and 6

    all_neurons_psth = make_raster(
        which_rois, dffs_ast, stim_dict_raster, img_VS_all, depth_list, frame_rate
    )
    np.save(target, all_neurons_psth)
    return all_neurons_psth


def get_recording_data(recording, flexilims_session, two_photon=True):

    processed = Path(flz.PARAMETERS["data_root"]["processed"])
    out = dict()

    # get vis stim
    with open(processed / recording.path / "img_VS.pickle", "rb") as handle:
        out["param_logger"] = pickle.load(handle)
    with open(processed / recording.path / "stim_dict.pickle", "rb") as handle:
        out["stim_dict"] = pickle.load(handle)

    # get suite2p data
    sess_ds = flz.get_children(
        parent_id=recording.origin_id,
        flexilims_session=flexilims_session,
        children_datatype="dataset",
    )
    suite2p_roi = sess_ds[sess_ds.dataset_type == "suite2p_rois"]
    assert len(suite2p_roi) == 1
    suite2p_roi = flz.Dataset.from_flexilims(
        data_series=suite2p_roi.iloc[0], flexilims_session=flexilims_session
    )
    out["ops"] = np.load(
        suite2p_roi.path_full / "suite2p" / "plane0" / "ops.npy", allow_pickle=True
    ).item()

    if not two_photon:
        # stop here with just behaviour
        return out

    # also load 2p data traces
    out["iscell"] = np.load(
        suite2p_roi.path_full / "suite2p" / "plane0" / "iscell.npy", allow_pickle=True
    )[:, 0]
    rec_ds = flz.get_children(
        parent_id=recording.id,
        flexilims_session=flexilims_session,
        children_datatype="dataset",
    )
    suite2p_traces = rec_ds[rec_ds.dataset_type == "suite2p_traces"]
    assert len(suite2p_traces) == 1
    suite2p_traces = flz.Dataset.from_flexilims(
        data_series=suite2p_traces.iloc[0], flexilims_session=flexilims_session
    )
    for datafile in suite2p_traces.path_full.glob("*.npy"):
        out[datafile.stem] = np.load(datafile, allow_pickle=True)

    # yiran reads it from session_protocol_folder

    project = [
        k
        for k, v in flz.PARAMETERS["project_ids"].items()
        if v == flexilims_session.project_id
    ][0]
    analysis_folder = (
        processed / project / "Analysis" / Path(*recording.genealogy) / "plane0"
    )
    import os

    os.listdir(analysis_folder)
    return out


def get_anova(
    which_rois, dffs_ast, depth_list, stim_dict, frame_rate, speed_arr, speed_thr_cal
):
    anova_ps = []
    for roi in which_rois:
        trace_arr, _ = process_params.create_trace_arr_per_roi(
            which_roi=roi,
            dffs=dffs_ast,
            depth_list=depth_list,
            stim_dict=stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        trace_arr[speed_arr < speed_thr_cal] = np.nan
        trace_arr_mean_eachtrial = np.nanmean(trace_arr, axis=2)
        groups = np.split(
            trace_arr_mean_eachtrial.reshape(len(depth_list), -1), len(depth_list)
        )
        groups = [x.ravel().tolist() for x in groups]
        _, p = scipy.stats.f_oneway(*groups)
        anova_ps.append(p)
    return np.array(anova_ps)


def cut_stim_dict(stim_dict, prop_fit, rng):
    """_summary_

    Args:
        stim_dict (_type_): _description_
        prop_fit (_type_): _description_
        rng (_type_): numpy random generator

    Returns:
        _type_: _description_
    """
    # cut the stim dict
    stim_dict_fit = {}
    stim_dict_raster = {}
    for depth, stims in stim_dict.items():
        ntrials = len(stims["start"])
        assert all([len(v) == ntrials for v in stims.values()])
        nfit = int(ntrials * prop_fit)
        randomised = np.arange(ntrials)  # we'll shuffle in place
        rng.shuffle(randomised)
        fit_trials = randomised[:nfit]
        raster_trials = randomised[nfit:]
        raster_dict = {}
        fit_dict = {}
        for k, v in stims.items():
            raster_dict[k] = v[fit_trials]
            fit_dict[k] = v[raster_trials]
        stim_dict_raster[depth] = raster_dict
        stim_dict_fit[depth] = fit_dict
    return stim_dict_fit, stim_dict_raster


def get_max_depth(which_rois, dffs_ast, depth_list, stim_dict):

    max_depths = np.ones(len(which_rois)) * 9999
    # index no. from depth_list indicating the max depth of each depth neuron
    max_depths_values = np.ones(len(which_rois)) * 9999
    # depth value = the max depth of each depth neuron

    for iroi, roi in enumerate(which_rois):
        trace_arr, _ = process_params.create_trace_arr_per_roi(
            roi,
            dffs_ast,
            depth_list,
            stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=15,
        )
        trace_arr_mean_eachtrial = np.nanmean(trace_arr, axis=2)
        trace_arr_depth_mean = np.nanmean(trace_arr_mean_eachtrial, axis=1)
        max_depth = np.where(trace_arr_depth_mean == np.max(trace_arr_depth_mean))[0]
        max_depths[iroi] = max_depth
        max_depths_values[iroi] = depth_list[max_depth[0]]

    max_depths[max_depths == 9999] = np.nan
    return max_depth


def gauss_fit(
    which_rois,
    dffs_ast,
    depth_list,
    stim_dict,
    frame_rate,
    speed_arr,
    max_depths,
    speed_thr_cal,
    batch_num,
    depth_min,
    depth_max,
    min_sigma,
):

    gaussian_depth_fit_df = pd.DataFrame(
        columns=[
            "ROI",
            "preferred_depth_idx",
            "a",
            "x0_logged",
            "log_sigma",
            "b",
            "r_sq",
        ]
    )
    gaussian_depth_fit_df.ROI = which_rois
    gaussian_depth_fit_df.preferred_depth_idx = max_depths

    def gaussian_func(x, a, x0, log_sigma, b):
        a = a
        sigma = np.exp(log_sigma) + min_sigma
        return (a * np.exp(-((x - x0) ** 2)) / (2 * sigma**2)) + b

    for iroi, roi in enumerate(which_rois):
        trace_arr, _ = process_params.create_trace_arr_per_roi(
            which_roi=roi,
            dffs=dffs_ast,
            depth_list=depth_list,
            stim_dict=stim_dict,
            mode="sort_by_depth",
            protocol="fix_length",
            blank_period=0,
            frame_rate=frame_rate,
        )
        trace_arr[speed_arr < speed_thr_cal] = np.nan
        trace_arr_mean_eachtrial = np.nanmean(trace_arr, axis=2)

        # When open loop, there will be nan values in this array because of low running speed sometimes
        trace_arr_mean_eachtrial = np.nan_to_num(trace_arr_mean_eachtrial)
        x = np.log(
            np.repeat(np.array(depth_list) * 100, trace_arr_mean_eachtrial.shape[1])
        )
        roi_number = np.where(which_rois == roi)[0][0]
        popt_arr = []
        r_sq_arr = []
        for ibatch in range(batch_num):
            np.random.seed(ibatch)
            p0 = np.concatenate(
                (
                    np.abs(np.random.normal(size=1)),
                    np.atleast_1d(
                        np.log(np.array(depth_list[int(max_depths[roi_number])]) * 100)
                    ),
                    np.abs(np.random.normal(size=1)),
                    np.random.normal(size=1),
                )
            ).flatten()
            popt, pcov = curve_fit(
                gaussian_func,
                x,
                trace_arr_mean_eachtrial.flatten(),
                p0=p0,
                maxfev=100000,
                bounds=(
                    [0, np.log(depth_min), 0, -np.inf],
                    [np.inf, np.log(depth_max), np.inf, np.inf],
                ),
            )

            y_pred = gaussian_func(x, *popt)
            r_sq = calculate_R_squared(trace_arr_mean_eachtrial.flatten(), y_pred)
            popt_arr.append(popt)
            r_sq_arr.append(r_sq)
        idx_best = np.argmax(r_sq_arr)
        popt_best = popt_arr[idx_best]
        rsq_best = r_sq_arr[idx_best]

        gaussian_depth_fit_df.iloc[iroi, 2:-1] = popt_best
        gaussian_depth_fit_df.iloc[iroi, -1] = rsq_best

        if iroi % 100 == 0:
            print(roi, flush=True)
    return gaussian_depth_fit_df


def make_raster(which_rois, dffs_ast, stim_dict, img_VS_all, depth_list, frame_rate):
    # create the distance array
    # Array of distance travelled from each trial start (m)
    all_neurons_psth = []
    distance_arr_original, _ = process_params.create_speed_arr(
        img_VS_all["EyeZ"],
        depth_list,
        stim_dict,
        mode="sort_by_depth",
        protocol="fix_length",
        blank_period=0,
        frame_rate=frame_rate,
    )
    for idepth in range(distance_arr_original.shape[0]):
        for itrial in range(distance_arr_original.shape[1]):
            distance_arr_original[idepth, itrial, :] = (
                distance_arr_original[idepth, itrial, :]
                - distance_arr_original[idepth, itrial, 0]
            )
    # not make psth for all rois and append to big list
    for roi in which_rois:
        binned_stats = get_PSTH(
            values=dffs_ast,
            dffs=dffs_ast,
            depth_list=depth_list,
            stim_dict=stim_dict,
            roi=roi,
            distance_arr=distance_arr_original,
            distance_bins=60,
            is_trace=True,
        )
        PSTH_ave = np.nanmean(binned_stats["binned_yrr"], axis=1).flatten()

        all_neurons_psth.append(PSTH_ave)
    return np.vstack(all_neurons_psth)
