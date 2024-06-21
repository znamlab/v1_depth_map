import os
import numpy as np
import pandas as pd
import flexiznam as flz
from cottage_analysis.pipelines import pipeline_utils


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
    for (key, item) in dict.items():
        if key in cols:
            if isinstance(item, float):
                df[key].iloc[index] = item
            elif isinstance(item, list):
                df[key] = create_nested_nan_list(np.array(item).ndim)
                df[key].iloc[index] = item
            elif isinstance(item, np.ndarray):
                df[key] = create_nested_nan_list(item.ndim)
                df[key].iloc[index]= item.tolist()
    return df


def find_columns_containing_string(df, substring):
    return [col for col in df.columns if substring in col]