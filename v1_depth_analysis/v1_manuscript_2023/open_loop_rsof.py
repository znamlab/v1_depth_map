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
import seaborn as sns
from scipy.stats import spearmanr

import flexiznam as flz
from cottage_analysis.preprocessing import synchronisation
from cottage_analysis.analysis import spheres, find_depth_neurons, common_utils, fit_gaussian_blob, size_control
from cottage_analysis.plotting import basic_vis_plots, plotting_utils
from cottage_analysis.pipelines import pipeline_utils


def plot_speeds_scatter(
    fig,
    neurons_df,
    xcol,
    ycol,
    xlabel="Running speed closedloop (cm/s)",
    ylabel="Running spped openloop (cm)",
    s=10,
    alpha=0.2,
    c='g',
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    cbar_width=0.01,
    plot_diagonal=False,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},):
    
    # Filter neurons_df
    neurons_df = neurons_df[(neurons_df["iscell"] == 1) & 
                            (neurons_df["depth_tuning_test_spearmanr_pval_closedloop"] < 0.05) &
                            (neurons_df["rsof_rsq_closedloop_g2d"] > 0.02) &
                            (neurons_df["rsof_rsq_openloop_actual_g2d"] > 0.02)                       
                            ]
    
    # Plot scatter
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height]) 
    X = neurons_df[xcol].values
    y = neurons_df[ycol].values
    ax.scatter(X, y, s=s, alpha=alpha, c=c,  edgecolors="none")
    r, p = spearmanr(X, y)
    if plot_diagonal:
        ax.plot(np.geomspace(X.min(),X.max(),1000), 
                np.geomspace(X.min(),X.max(),1000),
                'k',
                linestyle="dotted",
                linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=fontsize_dict["label"])
    ax.set_ylabel(ylabel, fontsize=fontsize_dict["label"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize_dict["tick"])
    ax.set_title(f"R = {r:.2f}, p = {p:.2e}", fontsize=fontsize_dict["title"])
    ax.set_aspect("equal")
    plotting_utils.despine()