import os
import numpy as np
import pandas as pd
import flexiznam as flz
from cottage_analysis.pipelines import pipeline_utils
from cottage_analysis.analysis import common_utils
from scipy import stats
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from tqdm import tqdm


def concatenate_all_neurons_df(
    flexilims_session,
    session_list,
    filename="neurons_df.pickle",
    cols=None,
    read_iscell=True,
    verbose=False,
):
    isess = 0
    for session in session_list:
        neurons_ds = pipeline_utils.create_neurons_ds(
            session_name=session,
            flexilims_session=flexilims_session,
            project=None,
            conflicts="skip",
        )
        if os.path.exists(neurons_ds.path_full.parent / filename):
            neurons_df = pd.read_pickle(neurons_ds.path_full.parent / filename)
            if isinstance(neurons_df, dict):
                neurons_df_temp = pd.DataFrame(columns=cols, index=[0])
                neurons_df = dict2df(neurons_df, neurons_df_temp, cols, 0)
            if (cols is None) or (set(cols).issubset(neurons_df.columns.tolist())):
                if cols is None:
                    neurons_df = neurons_df
                else:
                    neurons_df = neurons_df[cols]
                suite2p_ds = flz.get_datasets(
                    flexilims_session=flexilims_session,
                    origin_name=session,
                    dataset_type="suite2p_rois",
                    filter_datasets={"anatomical_only": 3},
                    allow_multiple=False,
                    return_dataseries=False,
                )
                if read_iscell:
                    iscell = np.load(
                        suite2p_ds.path_full / "plane0" / "iscell.npy",
                        allow_pickle=True,
                    )[:, 0]
                    neurons_df["iscell"] = iscell

                neurons_df["session"] = session
                if isess == 0:
                    neurons_df_all = neurons_df
                else:
                    neurons_df_all = pd.concat(
                        [neurons_df_all, neurons_df], ignore_index=True
                    )

                if verbose:
                    print(f"Finished concat {filename} from session {session}")
                isess += 1
            else:
                print(f"ERROR: SESSION {session}: specified cols not all in neurons_df")
        else:
            print(f"ERROR: SESSION {session}: {filename} not found")

    return neurons_df_all


def create_nested_nan_list(levels):
    nested_list = np.nan  # Start with np.nan
    for _ in range(levels):
        nested_list = [nested_list]  # Wrap the current structure in a new list
    return [nested_list]


def dict2df(dict, df, cols, index):
    for key, item in dict.items():
        if key in cols:
            if isinstance(item, float):
                df[key].iloc[index] = item
            elif isinstance(item, list):
                df[key] = create_nested_nan_list(1)
                df[key].iloc[index] = item
            elif isinstance(item, np.ndarray):
                df[key] = create_nested_nan_list(item.ndim)
                df[key].iloc[index] = item.tolist()
    return df


def find_columns_containing_string(df, substring):
    return [col for col in df.columns if substring in col]


def get_unique_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    return unique.values(), unique.keys()


def draw_axis_scalebars(
    ax,
    scalebar_x,
    scalebar_y,
    scalebar_width,
    scalebar_height,
    scalebar_labels,
    xlim=None,
    ylim=None,
    label_fontsize=5,
    linewidth=1,
    right=True,
    bottom=True,
):
    rect = patches.Rectangle(
        (scalebar_x, scalebar_y),
        scalebar_width,
        scalebar_height,
        linewidth=linewidth,
        edgecolor="none",
        facecolor="none",
    )
    ax.add_patch(rect)
    if right:
        right_edge = patches.FancyBboxPatch(
            (scalebar_x + scalebar_width, scalebar_y),
            0,
            scalebar_height,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(right_edge)
        ax.text(
            scalebar_x + scalebar_width * 1.2,
            scalebar_y + scalebar_height / 2,
            scalebar_labels[1],
            fontsize=label_fontsize,
            ha="left",
            va="center",
        )
    ax.set_ylim(ylim)
    if bottom:
        bottom_edge = patches.FancyBboxPatch(
            (scalebar_x, scalebar_y),
            scalebar_width,
            0,
            boxstyle="square,pad=0",
            linewidth=linewidth,
            edgecolor="black",
            facecolor="none",
        )
        ax.add_patch(bottom_edge)
        ax.text(
            scalebar_x,
            scalebar_y + scalebar_height * 0.1,
            scalebar_labels[0],
            fontsize=label_fontsize,
            ha="left",
            va="bottom",
        )
    ax.set_xlim(xlim)
    ax.axis("off")


def plot_white_rectangle(x0, y0, width, height):
    ax = plt.gcf().add_axes([x0, y0, width, height])
    # Define the rectangle's bottom-left corner, width, and height
    rectangle = patches.Rectangle((0, 0), 1, 1, edgecolor="white", facecolor="white")

    # Add the rectangle to the plot
    ax.add_patch(rectangle)

    # Set plot limits to better visualize the rectangle
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_aspect("equal")
    # fig.patch.set_facecolor('gray')


def ceil(a, base=1, precision=1):
    fold = a // (base * (10 ** (-precision)))
    extra = int((a % (base * (10 ** (-precision)))) > 0)
    ceiled_num = (fold + extra) * (base * (10 ** (-precision)))
    return np.round(ceiled_num, precision)


def hierarchical_bootstrap_stats(
    data,
    n_boots,
    xcol,
    resample_cols,
    ycol=None,
    correlation=False,
    difference=False,
    ratio=False,
):
    np.random.seed(0)
    if "mouse" not in data.columns:
        data["mouse"] = data["session"].str.split("_").str[0]
    distribution = np.zeros((n_boots, len(xcol)))
    r = np.zeros(len(xcol))
    if correlation:
        for icol, (x, y) in enumerate(zip(xcol, ycol)):
            r[icol] = stats.spearmanr(data[x], data[y])[0]
    elif difference:
        for icol, (x, y) in enumerate(zip(xcol, ycol)):
            r[icol] = np.median(data[x] - data[y])
    elif ratio:
        for icol, (x, y) in enumerate(zip(xcol, ycol)):
            r[icol] = np.median(data[x] / data[y])
    else:
        r = None
    for i in tqdm(range(n_boots)):
        sample = common_utils.bootstrap_sample(data, resample_cols)
        if ycol is None:
            for icol, x in enumerate(xcol):
                distribution[i, icol] = np.median(data.loc[sample][x])
        else:
            for icol, (x, y) in enumerate(zip(xcol, ycol)):
                if correlation:
                    distribution[i, icol] = stats.spearmanr(
                        data.loc[sample][x], data.loc[sample][y]
                    )[0]
                if difference:
                    distribution[i, icol] = np.median(
                        data.loc[sample][x] - data.loc[sample][y]
                    )
                if ratio:
                    distribution[i, icol] = np.median(
                        data.loc[sample][x] / data.loc[sample][y]
                    )
    plt.figure()
    for icol, x in enumerate(xcol):
        plt.subplot(2, len(xcol) // 2 + 1, icol + 1)
        plt.hist(distribution[:, icol], bins=31)
        plt.axvline(
            np.percentile(distribution[:, icol], 2.5), color="r", linestyle="--"
        )
        plt.axvline(
            np.percentile(distribution[:, icol], 97.5), color="r", linestyle="--"
        )
    return r, distribution


def calculate_pval_from_bootstrap(distribution, value):
    distribution = np.array(distribution)
    q_min = np.min([np.mean(distribution > value), np.mean(distribution < value)])
    return q_min * 2
