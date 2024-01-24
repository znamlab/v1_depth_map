import functools
print = functools.partial(print, flush=True)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42 # for pdfs
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import flexiznam as flz
from cottage_analysis.pipelines import pipeline_utils


def concatenate_all_neurons_df(flexilims_session, session_list, cols=None, read_iscell=True, verbose=False):
    for isess, session in enumerate(session_list):
        neurons_ds = pipeline_utils.create_neurons_ds(
        session_name=session,
        flexilims_session=flexilims_session,
        project=None,
        conflicts="skip",
        )
        neurons_df = pd.read_pickle(neurons_ds.path_full)
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
                iscell = np.load(suite2p_ds.path_full / "plane0" / "iscell.npy", allow_pickle=True)[:,0]
                neurons_df["iscell"] = iscell
            
            neurons_df["session"] = session
            if isess == 0:
                neurons_df_all = neurons_df
            else:   
                neurons_df_all = pd.concat([neurons_df_all, neurons_df], ignore_index=True)
                
            if verbose:
                print(f"Finished concat neurons_df from session {session}")
        else:
            print(f"ERROR: SESSION {session}: specified cols not all in neurons_df")
        
    return neurons_df_all