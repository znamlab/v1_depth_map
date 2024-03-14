import functools

print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.rcParams["pdf.fonttype"] = 42  # for pdfs
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import scipy
from scipy.stats import ttest_rel
import itertools

import flexiznam as flz
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import plotting_utils, depth_decoder_plots
from cottage_analysis.analysis import common_utils


def concatenate_all_decoder_results(
    flexilims_session, session_list, filename="decoder_results.pickle"
):
    isess = 0
    results = pd.DataFrame(
        columns=[
            "session",
            "accuracy_closedloop",
            "accuracy_openloop",
            "conmat_closedloop",
            "conmat_openloop",
        ]
    )
    results["conmat_closedloop"] = [[np.nan]]
    results["conmat_openloop"] = [[np.nan]]

    for session in session_list:
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )

        filepath = neurons_ds.path_full.parent / filename
        if filepath.is_file():
            decoder_dict = pd.read_pickle(neurons_ds.path_full.parent / filename)
            results.at[isess, "session"] = session
            results.at[isess, "accuracy_closedloop"] = decoder_dict[
                "accuracy_closedloop"
            ]
            results.at[isess, "accuracy_openloop"] = decoder_dict["accuracy_openloop"]
            results.at[isess, "conmat_closedloop"] = decoder_dict["conmat_closedloop"]
            results.at[isess, "conmat_openloop"] = decoder_dict["conmat_openloop"]
            isess += 1
            print(f"SESSION {session}: decoder_results concatenated")
        else:
            print(f"ERROR: SESSION {session}: decoder_results not found")
            continue
    return results


def bar_plot_ttest(
    group1, group2, labels, fig, plot_x, plot_y, plot_width, plot_height
):
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    # Calculate means and standard deviations
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1), np.std(group2)

    # Calculate t-test
    t_statistic, p_value = ttest_rel(group1, group2)

    # Plot the data
    bar1 = ax.bar(0, mean1, yerr=std1, capsize=10, width=0.25, label=labels[0])
    bar2 = ax.bar(0.3, mean2, yerr=std2, capsize=10, width=0.25, label=labels[1])

    # Add bracket and asterisk if significant
    if p_value < 0.05:
        star = "*"
        if p_value < 0.01:
            star = "**"
        elif p_value < 0.001:
            star = "***"
        # Add square bracket
        x0, x1 = (
            bar1[0].get_x() + bar1[0].get_width() / 2,
            bar2[0].get_x() + bar2[0].get_width() / 2,
        )
        y, h, col = max([mean1 + std1]), 0.05, "k"
        ax.plot([x0, x0, x1, x1], [y + h, y + 2 * h, y + 2 * h, y + h], lw=1.5, c=col)
        # Add asterisk on top of bracket
        ax.text((x0 + x1) * 0.5, y + 2 * h, star, ha="center", va="bottom", fontsize=24)

    # Set the x-axis labels and title
    ax.set_xticks([0, 0.3])
    ax.set_xticklabels([labels[0], labels[1]])

    plotting_utils.despine()


def dot_plot(
    group1,
    group2,
    labels,
    baselines,
    fig,
    group3=None,
    group4=None,
    markersize=5,
    colors=["b", "g"],
    errorbar=False,
    errorbar_colors=["r","r"],
    linewidth=4,
    ylim=[0, 1],
    xlabel=["Closed loop", "Open loop"],
    ylabel="Classification accuracy",
    plot_x=0,
    plot_y=1,
    plot_width=1,
    plot_height=1,
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
):
    ax = fig.add_axes([plot_x, plot_y, plot_width, plot_height])
    ax.plot(
        [0.3, 0.7],
        [group1, group2],
        f"{colors[0]}o-",
        alpha=0.7,
        label=labels[0],
        markersize=markersize,
        linewidth=linewidth,
    )
    if errorbar:
        for group, x in zip([group1, group2],[0.3,0.7]):
            group_mean = np.mean(group)
            group_ci = common_utils.get_bootstrap_ci(group, sig_level=0.05, n_bootstraps=1000, func=np.nanmean)
            ax.errorbar(x, group_mean, yerr=(group_ci[1]-group_mean), fmt='o', color=errorbar_colors[0], markersize=markersize, linewidth=linewidth)
    
    if len(baselines) > 0:
        ax.axhline(y=baselines[0], color=colors[0], linestyle="dotted", linewidth=linewidth)
    t_statistic, p_value = ttest_rel(group1, group2)
    print(f"t-test for {labels[0]}: {p_value}")
    if group3 is not None:
        ax.plot(
            [0.3, 0.7],
            [group3, group4],
            f"{colors[1]}o-",
            alpha=0.7,
            label=labels[1],
            markersize=markersize,
            linewidth=linewidth,
        )
        if len(baselines) > 0:
            ax.axhline(
                y=baselines[1], color=colors[1], linestyle="dotted", linewidth=linewidth
            )
        t_statistic, p_value = ttest_rel(group3, group4)
        print(f"t-test for {labels[1]}: {p_value}")
        
        if errorbar:
            for group, x in zip([group3, group4],[0.3,0.7]):
                group_mean = np.mean(group)
                group_ci = common_utils.get_bootstrap_ci(group, sig_level=0.05, n_bootstraps=1000, func=np.nanmean)
                ax.errorbar(x, group_mean, yerr=(group_ci[1]-group_mean), fmt='o', color=errorbar_colors[0], markersize=markersize, linewidth=linewidth)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.xticks([0.3, 0.7], fontsize=fontsize_dict["tick"])
    plt.yticks(fontsize=fontsize_dict["tick"])
    plt.xlim([0.1, 0.9])
    plt.ylim(ylim)
    ax.set_xticklabels(
        xlabel, fontsize=fontsize_dict["label"]
    )
    plt.ylabel(ylabel, fontsize=fontsize_dict["label"])
    # ax.legend(fontsize=fontsize_dict['legend'])


def calculate_average_confusion_matrix(results, conmat_column, sfx):
    results[f"conmat_prop{sfx}"] = results[f"{conmat_column}{sfx}"].apply(
        lambda x: x / np.sum(x)
    )
    conmat_mean = np.mean(np.stack(results[f"conmat_prop{sfx}"]), axis=0)
    return results, conmat_mean


def plot_confusion_matrix(
    conmat,
    ax,
    vmax,
    fontsize_dict,
):
    ax.imshow(conmat, interpolation="nearest", cmap="Blues", vmax=vmax, vmin=0)
    fmt = "d"
    thresh = vmax / 2.0
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        display = conmat[i, j]
        if display > 1:
            display = f"{int(display)}"
        else:
            display = f"{display:.2f}"
        plt.text(
            j,
            i,
            display,
            horizontalalignment="center",
            color="white" if conmat[i, j] > thresh else "black",
            fontsize=fontsize_dict["legend"],
        )
    plt.xlabel("Predicted depth class", fontsize=fontsize_dict["label"])
    plt.ylabel("True depth class", fontsize=fontsize_dict["label"])


def plot_closed_open_conmat(
    conmat_closed,
    conmat_open,
    normalize,
    fig,
    plot_x,
    plot_y,
    plot_width,
    plot_height,
    fontsize_dict,
):
    if normalize:
        conmat_closed = conmat_closed / conmat_closed.sum(axis=1)[:, np.newaxis]
        conmat_open = conmat_open / conmat_open.sum(axis=1)[:, np.newaxis]
    vmax = conmat_closed.max()
    ax = fig.add_axes([plot_x, plot_y, plot_width / 2 * 0.8, plot_height * 0.8])
    plot_confusion_matrix(conmat_closed, ax, vmax, fontsize_dict)
    ax.set_title("Closed loop", fontsize=fontsize_dict["title"])
    ax = fig.add_axes(
        [plot_x + plot_width / 2, plot_y, plot_width / 2 * 0.8, plot_height * 0.8]
    )
    plot_confusion_matrix(conmat_open, ax, vmax, fontsize_dict)
    ax.set_title("Open loop", fontsize=fontsize_dict["title"])


def make_error_weight_matrix(size):
    # predict depth - true depth
    matrix = np.zeros((size, size))  # Initialize a size x size matrix of zeros
    for i in range(size):
        for j in range(size):
            if i < j:
                matrix[i, j] = 1  # Set upper half to 1, as predicted depth class is larger than true depth class
            elif i > j:
                matrix[i, j] = -1   # Set lower half to -1, as predicted depth class is smaller than true depth class
    return matrix


def calculate_error(conmat):
    # calculate average squared error of log(depth)
    if conmat.shape[0] == 5:
        depth_list = np.geomspace(6,600,5)
    elif conmat.shape[0] == 8:
        depth_list = np.geomspace(5,640,8)
    log_distance = np.log(depth_list[1]) - np.log(depth_list[0])
    error_weight_matrix = make_error_weight_matrix(conmat.shape[0])
    error = np.sum(np.square(np.multiply(conmat,error_weight_matrix) * log_distance)) / np.sum(conmat)
    return error


def calculate_error_all_sessions(results):
    results["error_closedloop"] = results["conmat_closedloop"].apply(calculate_error)
    results["error_openloop"] = results["conmat_openloop"].apply(calculate_error)
    return results