from pathlib import Path
import flexiznam as flz
import pandas as pd
import yaml
from flexilims.offline import download_database
from flexiznam.schema import Dataset
from v1_depth_analysis.config import MICE, PROJECT
from tifffile import TiffFile

FLM_SESS = flz.get_flexilims_session(project_id=PROJECT)


def get_sessions(mice=None, flm_sess=FLM_SESS):
    """Get recording sessions from flexilims

    Args:
        mice (list, optional): List of mice to consider. If None will load all mice
        flm_sess (flz.Session, optional): Flexilims session to interact with database.
            Defaults to FLM_SESS.

    Returns:
        list: List of session series loaded from flexilims. Will contain only session
            of mice defined in config.MICE
    """
    if mice is None:
        mice = MICE
    raw_path = Path(flz.PARAMETERS["data_root"]["raw"])
    if isinstance(mice, str):
        mice = [mice]

    sessions_list = []
    for mouse in mice:
        mouse_folder = raw_path / PROJECT / mouse
        assert mouse_folder.is_dir(), f"Folder {mouse_folder} does not exist"
        mouse_entity = flz.get_entity(
            name=mouse, datatype="mouse", flexilims_session=flm_sess
        )
        sessions = flz.get_children(
            mouse_entity.id, children_datatype="session", flexilims_session=flm_sess
        )
        sessions_list.append(sessions)
    return pd.concat(sessions_list)


def get_recordings(protocol="SpheresPermTubeReward", sessions=None, flm_sess=FLM_SESS):
    """Get a list of recordings with a given protocol

    Args:
        protocol (str, optional): Protocol to keep. Defaults to "SpheresPermTubeReward".
        sessions (list, optional): List of session to consider. If None will load all
            sessions defiend in config.SESSION. Defaults to None.
        flm_sess (flz.Session, optional): Flexilims session. Defaults to FLM_SESS.

    Returns:
        list: List of recordings series loaded from flexilims
    """
    if sessions is None:
        session = get_sessions(MICE, flm_sess=flm_sess)
    recordings = []
    for sess_name, sess in session.iterrows():
        recs = flz.get_children(
            sess.id, children_datatype="recording", flexilims_session=flm_sess
        )
        for _, rec in recs.iterrows():
            if rec["protocol"] == protocol:
                recordings.append(rec)
    return recordings


def get_datasets(
    recordings, dataset_type=None, dataset_name_contains=None, flm_sess=FLM_SESS
):
    """Get a list of datasets from a recording list

    Args:
        recordings (list): List of recordings as produced by `get_recordings` or single
            recording series
        dataset_type (str, optional): If not None, return only datasets of type
            `dataset_type`. Defaults to None.
        dataset_name_contains (str, optional): If not None, return only datasets whose
            name contains `dataset_name_contains`. Defaults to None.
        flm_sess (flz.SESSION, optional): Flexilims session. Defaults to FLM_SESS.

    Returns:
        list: List of flz.schema.Dataset objects
    """
    if isinstance(recordings, pd.Series):
        recordings = [recordings]

    all_datasets = []
    for rec in recordings:
        datasets = flz.get_children(
            rec.id, children_datatype="dataset", flexilims_session=flm_sess
        )
        datasets = [
            Dataset.from_flexilims(data_series=ds, flexilims_session=flm_sess)
            for _, ds in datasets.iterrows()
        ]
        if dataset_type is not None:
            datasets = [ds for ds in datasets if ds.dataset_type == dataset_type]
        if dataset_name_contains is not None:
            datasets = [
                ds for ds in datasets if dataset_name_contains in ds.dataset_name
            ]
        all_datasets.extend(datasets)
    return all_datasets


def download_full_flexilims_database(flexilims_session, target_file=None):
    """Download the full flexilims database as json and save to file

    Args:
        flexilims_session (flz.Session): Flexilims session
        target_file (str, optional): Path to save json file. Defaults to None.

    Returns:
        dict: The json data
    """

    json_data = download_database(
        flexilims_session, root_datatypes=("mouse"), verbose=True
    )
    if target_file is not None:
        with open(target_file, "w") as f:
            yaml.dump(json_data, f)
    return json_data


def get_si_metadata(flexilims_session, session):
    recording = flz.get_children(
        parent_name=session,
        flexilims_session=flexilims_session,
        children_datatype="recording",
    ).iloc[0]
    dataset = flz.get_children(
        parent_name=recording["name"],
        flexilims_session=flexilims_session,
        children_datatype="dataset",
        filter={"dataset_type": "scanimage"},
    ).iloc[0]
    data_root = flz.get_data_root("raw", flexilims_session=flexilims_session)
    tif_path = data_root / recording["path"] / sorted(dataset["tif_files"])[0]
    return TiffFile(tif_path).scanimage_metadata
