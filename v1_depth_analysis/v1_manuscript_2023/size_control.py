import numpy as np
import pandas as pd
import scipy
import flexiznam as flz

from cottage_analysis.plotting import plotting_utils


def plot_depth_size_fit_comparison(
    fig,
    neurons_df,
    filter=None,
    use_cols={
        "depth_fit_r2": "depth_tuning_test_rsq_closedloop",
        "size_fit_r2": "size_tuning_test_rsq_closedloop",
        "depth_fit_pval": "depth_tuning_test_spearmanr_pval_closedloop",
        "size_fit_pval": "size_tuning_test_spearmanr_pv,al_closedloop",
    },
    plot_type="scatter",
    s=5,
    c="k",
    alpha=0.5,
    plot_x=0,
    plot_y=0,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10},):
    
    # Plot scatter of r2 depth vs. size
    if filter is None:
        filtered_neurons_df = neurons_df
    else:
        filtered_neurons_df = neurons_df[filter]
        
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    
    if plot_type == "scatter":
        ax.scatter(filtered_neurons_df[use_cols["depth_fit_r2"]], filtered_neurons_df[use_cols["size_fit_r2"]], s=s, c=c, alpha=alpha, edgecolors="none")
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], c="k", linestyle="dotted", linewidth=1)
        ax.set_aspect("equal")
        
        ax.set_xlabel("Depth fit r-squared", fontsize=fontsize_dict["label"])
        ax.set_ylabel("Size fit r-squared", fontsize=fontsize_dict["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        
    elif plot_type == "hist":
        diff = filtered_neurons_df[use_cols["depth_fit_r2"]] - filtered_neurons_df[use_cols["size_fit_r2"]]
        ax.hist(diff, bins=30, color=c, alpha=alpha)
        ax.set_xlabel("Difference between depth and \nsize tuning r-squared", fontsize=fontsize_dict["label"])
        print(f"median {np.median(diff)}")
    
    plotting_utils.despine()  
    print(scipy.stats.wilcoxon(filtered_neurons_df[use_cols["depth_fit_r2"]], filtered_neurons_df[use_cols["size_fit_r2"]]))