import functools
print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # for pdfs
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import pickle
from tqdm import tqdm
import scipy

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import spheres, find_depth_neurons, common_utils, fit_gaussian_blob, size_control
from cottage_analysis.plotting import basic_vis_plots, plotting_utils
from cottage_analysis.pipelines import pipeline_utils


def plot_stimuli_frame(fig,
                       frames, 
                       iframe, 
                       idepth,
                       ndepths,
                       plot_x=0,
                       plot_y=1,
                       plot_width=1,
                       plot_height=1,
                       plot_prop=1,
                       fontsize_dict={'title': 15, 'label': 10, 'tick': 10}):
    for i in range(ndepths):
        ax = fig.add_axes([plot_x, plot_y-plot_height/ndepths*i, plot_width, plot_height/ndepths*plot_prop])
        if i == idepth:
            this_frame = frames[iframe].astype(float)
            this_frame[this_frame==0] = 0.5
            ax.imshow(this_frame, cmap='gray_r', origin='lower',extent=[0,120,-40,40], aspect='equal', vmax=1, vmin=0)
        else:
            ax.imshow(np.ones_like(frames[iframe])*0.5, cmap='gray_r', origin='lower',extent=[0,120,-40,40], aspect='equal', vmax=1, vmin=0)    
        if i == ndepths-1:
            ax.set_xlabel('Azimuth (deg)', fontsize=fontsize_dict['label'])
        elif i == ndepths//2:
            ax.set_ylabel('Elevation (deg)', fontsize=fontsize_dict['label'])
        if i != ndepths-1:
            ax.set_xticklabels([])
        ax.tick_params(axis='both', labelsize=fontsize_dict['tick'])
        

def plot_rf(
    fig,
    neurons_df,
    roi, 
    ndepths=8,
    frame_shape=(16,24),
    plot_x=0,
    plot_y=1,
    plot_width=1,
    plot_height=1,
    plot_prop=0.9,
    xlabel='Azimuth (deg)',
    ylabel='Elevation (deg)',
    fontsize_dict={'title': 15, 'label': 10, 'tick': 5},
):
    coef = neurons_df.loc[roi, 'rf_coef'][:,:-1]
    coef = coef.reshape(coef.shape[0], ndepths, frame_shape[0], frame_shape[1])
    coef_mean = np.mean(coef, axis=0)
    coef_max = np.nanmax(coef_mean)
    
    for i in range(ndepths):
        ax = fig.add_axes([plot_x, plot_y-plot_height/ndepths*i, plot_width, plot_height/ndepths*plot_prop])
        plt.imshow(coef_mean[i,:,:], 
                   origin='lower', 
                   cmap='bwr', 
                   extent=[0,120,-40,40],
                   vmin=-coef_max, 
                   vmax=coef_max)
        if i != ndepths-1:
            plt.gca().set_xticklabels([])
        if i == ndepths//2:
            ax.set_ylabel(ylabel, fontsize=fontsize_dict['label'])
        if i == ndepths-1:
            ax.set_xlabel(xlabel, fontsize=fontsize_dict['label'])
        ax.tick_params(axis='both', labelsize=fontsize_dict['tick'])
        

def get_rf_results(project, sessions):
    for i, session_name in enumerate(sessions):
        flexilims_session = flz.get_flexilims_session(project_id=project)
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session_name,
            flexilims_session=flexilims_session,
            project=project,
            conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        results = pd.DataFrame(
            {
                "session": np.nan,
                "roi": np.nan,
                "iscell": np.nan,
                "preferred_depth": np.nan,
                "preferred_depth_rsq": np.nan,
                "coef": [[np.nan]]*len(neurons_df),
                "coef_ipsi": [[np.nan]]*len(neurons_df),
            }
        )
        
        # Add roi, preferred depth, iscell to results
        results["roi"] = np.arange(len(neurons_df))
        results["session"] = session_name
        results["preferred_depth"] = neurons_df["preferred_depth_closedloop"]
        results["preferred_depth_rsq"] = neurons_df["depth_tuning_test_rsq_closedloop"]
        exp_session = flz.get_entity(
            datatype="session", name=session_name, flexilims_session=flexilims_session
        )
        suite2p_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=exp_session.name,
            dataset_type="suite2p_rois",
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            return_dataseries=False,
        )
        iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[:,0]
        results["iscell"] = iscell
        
        # Add coef to results
        results["rf_coef"] = neurons_df["rf_coef"]
        results["rf_coef_ipsi"] = neurons_df["rf_coef_ipsi"]
        
        if i==0:
            results_all = results
        else:
            results_all = pd.concat([results_all, results], axis=0, ignore_index=True)
            
    return results_all


def plot_rf_centers(fig,
                    results, 
                    colors=['r','b'],
                    ndepths=8,
                    frame_shape=(16,24),
                    n_stds=5,
                    plot_x=0,
                    plot_y=1,
                    plot_width=1,
                    plot_height=1,
                    fontsize_dict={'title': 15, 'label': 10, 'tick': 5},
                    ):
    ax=fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    sessions = results.session.unique()
    
    for i in range(len(sessions)):
        # Get the coef and ipsi+_coef from each session
        session = sessions[i]
        results_sess = results[results.session==session]
        coef = np.stack(results_sess.rf_coef.values)
        coef_ = (coef[:,:,:-1]).reshape(coef.shape[0], coef.shape[1], ndepths, frame_shape[0], frame_shape[1])
        coef_mean = np.mean(np.mean(coef_, axis=1), axis=1)

        coef_ipsi = np.stack(results_sess.rf_coef_ipsi.values)
        coef_ipsi_ = (coef_ipsi[:,:,:-1]).reshape(coef_ipsi.shape[0], coef.shape[1], ndepths, frame_shape[0], frame_shape[1])
        coef_ipsi_mean = np.mean(np.mean(coef_ipsi_, axis=1), axis=1)

        # Find the center (index of maximum value of fitted RF)
        max_idx = [[np.unravel_index(coef_mean[i, :, :].argmax(), coef_mean[0, :, :].shape)[0],
                    np.unravel_index(coef_mean[i, :, :].argmax(), coef_mean[0, :, :].shape)[1]] for i in range(coef_mean.shape[0])]
        max_idx = np.array(max_idx)

        def index_to_deg(idx,resolution=5, jitter=True, n_ele=80, n_azi=120):
            azi = idx[:,1]*resolution  
            ele = idx[:,0]*resolution-n_ele/2
            if jitter:
                azi = azi + np.random.normal(0,1,size=azi.shape)
                ele = ele + np.random.normal(0,1,size=ele.shape)
            return azi, ele
        azi, ele = index_to_deg(max_idx)
        
        # Find cells with significant RF
        sig, sig_ipsi = spheres.find_sig_rfs(np.swapaxes(np.swapaxes(coef, 0, 2),0,1), 
                                             np.swapaxes(np.swapaxes(coef_ipsi, 0, 2),0,1),  
                                             n_std=n_stds)

        # Plot
        ax.scatter(azi[sig & (results_sess.iscell==1)], ele[sig & (results_sess.iscell==1)], c=colors[i], edgecolors='none', s=10, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        plotting_utils.despine()
        ax.set_xlabel('Azimuth (deg)', fontsize=fontsize_dict['label'])
        ax.set_ylabel('Elevation (deg)', fontsize=fontsize_dict['label'])
        ax.set_xlim([0,120])
        ax.set_ylim([-40,40])
        ax.tick_params(axis='both', labelsize=fontsize_dict['tick'])
        
        
def load_sig_rf(flexilims_session, session_list, depth_neuron_thr=0.04, n_std=5):
    all_sig = []
    all_sig_ipsi = []
    for isess, session in enumerate(session_list):
        # Load neurons_df
        neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session,
        flexilims_session=flexilims_session,
        project=None,
        conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
        
        # Load iscell
        suite2p_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=session,
            dataset_type="suite2p_rois",
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            return_dataseries=False,
            )   
        iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[:,0]
        neurons_df["iscell"] = iscell
        
        # Load RF significant % 
        coef = np.stack(neurons_df["rf_coef"].values)
        coef_ipsi = np.stack(neurons_df["rf_coef_ipsi"].values)
        sig, sig_ipsi = spheres.find_sig_rfs(np.swapaxes(np.swapaxes(coef, 0, 2),0,1), 
                                            np.swapaxes(np.swapaxes(coef_ipsi, 0, 2),0,1),  
                                            n_std=n_std)
        select_neurons = (neurons_df["iscell"]==1) & (neurons_df["depth_tuning_test_rsq_closedloop"]>depth_neuron_thr)
        sig = sig[select_neurons]
        sig_ipsi = sig_ipsi[select_neurons]
        all_sig.append(np.mean(sig))
        all_sig_ipsi.append(np.mean(sig_ipsi))
        
    return all_sig, all_sig_ipsi


def plot_sig_rf_perc(fig,
                     all_sig,
                     all_sig_ipsi,
                     bar_color='k',
                     scatter_color='k',
                     scatter_size=10,
                     plot_x=0,
                     plot_y=1,
                     plot_width=1,
                     plot_height=1,
                     fontsize_dict={'title': 15, 'label': 10, 'tick': 5}
):
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    ax.bar(x=[0,1], 
           height=[np.mean(all_sig), np.mean(all_sig_ipsi)], 
           yerr=[scipy.stats.sem(all_sig), scipy.stats.sem(all_sig_ipsi)], 
           capsize=10,
           color=bar_color, 
           alpha=0.5)
    ax.scatter(x=np.zeros(len(all_sig)), 
               y=all_sig, 
               color=scatter_color, 
               s=scatter_size)
    ax.scatter(x=np.ones(len(all_sig_ipsi)), 
               y=all_sig_ipsi, 
               color=scatter_color, 
               s=scatter_size)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Contra-\nlateral', 'Ipsi-\nlateral'], fontsize=fontsize_dict['label'])
    ax.set_ylabel('Proportion of neurons with \nsignificant receptive field', fontsize=fontsize_dict['label'])
    ax.set_ylim([0,1])
    plotting_utils.despine()
    ax.tick_params(axis='y', which='major', labelsize=fontsize_dict['tick'])
    plt.tight_layout()