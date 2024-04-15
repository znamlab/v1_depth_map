import numpy as np
import pandas as pd
import matplotlib.patheffects as PathEffects

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import itertools

from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.plotting import plotting_utils


def concatenate_all_decoder_results(
    flexilims_session, session_list, filename="decoder_results.pickle"
):
    all_sessions = []
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
            decoder_dict["ndepths"] = len(decoder_dict["conmat_closedloop"])
            decoder_dict["session"] = session
            print(f"SESSION {session}: decoder_results concatenated")
            all_sessions.append(decoder_dict)
        else:
            print(f"ERROR: SESSION {session}: decoder_results not found")
            continue
    results = pd.DataFrame(all_sessions)
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


def decoder_accuracy(
    decoder_results,
    markersize=5,
    colors=["b", "g"],
    linewidth=0.5,
    xlabel=["Closed loop", "Open loop"],
    ylabel="Classification accuracy",
    fontsize_dict={"title": 15, "label": 10, "tick": 10, "legend": 5},
    mode="accuracy"
):
    ndepths_list = decoder_results["ndepths"].unique()
    for ndepths, color in zip(ndepths_list, colors):
        this_ndepths = decoder_results[decoder_results["ndepths"] == ndepths]
        plt.plot(
            [1, 2],
            [this_ndepths[f"{mode}_closedloop"], this_ndepths[f"{mode}_openloop"]],
            f"{color}o-",
            alpha=0.7,
            label=f"{ndepths} depths",
            markersize=markersize,
            linewidth=linewidth,
        )
        if mode == "accuracy":
            plt.axhline(y=1 / ndepths, color=color, linestyle="dashed", linewidth=linewidth)
        plt.plot(
            [1, 2],
            [
                np.median(this_ndepths[f"{mode}_closedloop"]),
                np.median(this_ndepths[f"{mode}_openloop"]),
            ],
            f"{color}-",
            alpha=0.7,
            markersize=markersize,
            linewidth=linewidth * 2,
        )
        for icol, col in enumerate([f"{mode}_closedloop", f"{mode}_openloop"]):
            plt.plot(
                [icol + 0.8, icol + 1.2],
                [np.median(this_ndepths[col]), np.median(this_ndepths[col])],
                color,
                lw=2,
            )
    _, p_value = wilcoxon(
        decoder_results[f"{mode}_closedloop"], decoder_results[f"{mode}_openloop"]
    )
    print(f"p-value: {p_value}")
    plotting_utils.despine()
    plt.xticks([1, 2], xlabel, fontsize=fontsize_dict["label"], rotation=45, ha="right")
    plt.yticks(fontsize=fontsize_dict["tick"])
    plt.xlim([0.5, 2.5])
    if mode == "accuracy":
        plt.ylim([0, 1])
    plt.ylabel(ylabel, fontsize=fontsize_dict["label"])


def calculate_average_confusion_matrix(decoder_results):
    conmat_mean = {}
    for sfx in ["closedloop", "openloop"]:
        decoder_results[f"conmat_prop_{sfx}"] = decoder_results[f"conmat_{sfx}"].apply(
            lambda x: x / np.sum(x)
        )
        conmat_mean[sfx] = np.mean(
            np.stack(decoder_results[f"conmat_prop_{sfx}"]), axis=0
        )
    return conmat_mean


def plot_confusion_matrix(
    conmat,
    ax,
    vmax,
    fontsize_dict,
    depths=np.logspace(np.log2(5), np.log2(640), 8, base=2),
):
    im = ax.imshow(conmat, interpolation="nearest", cmap="magma", vmax=vmax, vmin=0)
    for i, j in itertools.product(range(conmat.shape[0]), range(conmat.shape[1])):
        display = conmat[i, j]
        if display > 1:
            display = f"{int(display)}"
        else:
            display = f"{display:.2f}"
        # plt.text(
        #     j,
        #     i,
        #     display,
        #     horizontalalignment="center",
        #     color="black" if conmat[i, j] > vmax / 2 else "white",
        #     fontsize=fontsize_dict["tick"],
        # )
    plt.xticks(np.arange(len(depths)), depths, fontsize=fontsize_dict["tick"])
    plt.yticks(np.arange(len(depths)), depths, fontsize=fontsize_dict["tick"])
    plt.xlabel("Predicted virtual\ndepth (cm)", fontsize=fontsize_dict["label"])
    plt.ylabel("True virtual depth (cm)", fontsize=fontsize_dict["label"])
    return im


def plot_closed_open_conmat(
    conmat_mean,
    normalize,
    fig,
    plot_x,
    plot_y,
    plot_width,
    plot_height,
    fontsize_dict,
):
    conmat_closed = conmat_mean["closedloop"]
    conmat_open = conmat_mean["openloop"]
    if normalize:
        conmat_closed = conmat_closed / conmat_closed.sum(axis=1)[:, np.newaxis]
        conmat_open = conmat_open / conmat_open.sum(axis=1)[:, np.newaxis]
    vmax = conmat_closed.max()
    ax = fig.add_axes([plot_x, plot_y, plot_width / 2 * 0.8, plot_height * 0.8])
    if len(conmat_closed) == 8:
        depths = np.logspace(np.log2(5), np.log2(640), 8, base=2)
    else:
        depths = np.logspace(np.log2(6), np.log2(600), 5, base=2)
    depths = np.round(depths).astype(int)
    plot_confusion_matrix(conmat_closed, ax, vmax, fontsize_dict, depths=depths)
    ax.set_title("Closed loop", fontsize=fontsize_dict["label"])
    ax = fig.add_axes(
        [plot_x + plot_width / 2, plot_y, plot_width / 2 * 0.8, plot_height * 0.8]
    )
    im = plot_confusion_matrix(conmat_open, ax, vmax, fontsize_dict, depths=depths)
    ax.set_title("Open loop", fontsize=fontsize_dict["label"])
    ax.set_yticks([])
    ax.set_ylabel("")
    bounds = ax.get_position().bounds
    cbar_ax = fig.add_axes([bounds[0] + bounds[2] + 0.01, bounds[1], 0.01, bounds[3] / 2])
    fig.colorbar(
        ax=ax,
        mappable=im,
        cax=cbar_ax,
    )
    cbar_ax.tick_params(labelsize=fontsize_dict["tick"])


def calculate_error(conmat):
    ndepths = conmat.shape[0]
    m = np.repeat(np.arange(ndepths)[np.newaxis, :], ndepths, axis=0)

    errs = np.abs(m - m.T).flatten()

    mean_error = np.sum(conmat.flatten() * errs) / np.sum(
        conmat.flatten()
    )
    if ndepths == 5:
        mean_error = mean_error * np.log2(np.sqrt(10))
    return mean_error


def calculate_error_all_sessions(decoder_results):
    decoder_results["error_closedloop"] = decoder_results["conmat_closedloop"].apply(calculate_error)
    decoder_results["error_openloop"] = decoder_results["conmat_openloop"].apply(calculate_error)
    return decoder_results
