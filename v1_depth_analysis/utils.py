from pathlib import Path
import flexiznam as flm
from flexiznam.schema import Dataset
from v1_depth_analysis.config import SESSIONS, PROJECT

FLM_SESS = flm.get_flexilims_session(project_id=PROJECT)


def get_sessions(flm_sess=FLM_SESS):
    """Get recording sessions from flexilims

    Args:
        flm_sess (flm.Session, optional): Flexilims session to interact with database.
            Defaults to FLM_SESS.

    Returns:
        list: List of session series loaded from flexilims. Will contain only session
            defined in config.SESSIONS
    """
    raw_path = Path(flm.PARAMETERS["data_root"]["raw"])

    sessions = []
    for mouse, sessions in SESSIONS.items():
        mouse_folder = raw_path / PROJECT / mouse
        assert mouse_folder.is_dir()
        for session in sessions:
            session_folder = mouse_folder / f"S20{session}"
            assert session_folder.is_dir()
            sess = flm.get_entity(
                name=f"{mouse}_S20{session}", flexilims_session=flm_sess
            )
            sessions.append(sess)
    return sessions


def get_recordings(protocol="SpheresPermTubeReward", sessions=None, flm_sess=FLM_SESS):
    """Get a list of recordings with a given protocol

    Args:
        protocol (str, optional): Protocol to keep. Defaults to "SpheresPermTubeReward".
        sessions (list, optional): List of session to consider. If None will load all
            sessions defiend in config.SESSION. Defaults to None.
        flm_sess (flm.Session, optional): Flexilims session. Defaults to FLM_SESS.

    Returns:
        list: List of recordings series loaded from flexilims
    """
    if sessions is None:
        session = get_sessions(flm_sess=flm_sess)
    recordings = []
    for sess in session:
        recs = flm.get_children(
            sess.id, children_datatype="recording", flexilims_session=flm_sess
        )
        for rec_name, rec in recs.iterrows():
            if rec["protocol"] == protocol:
                recordings.append(rec)
    return recordings


def get_datasets(
    recordings, dataset_type=None, dataset_name_contains=None, flm_sess=FLM_SESS
):
    """Get a list of datasets from a recording list

    Args:
        recordings (list): List of recordings as produced by `get_recordings`
        dataset_type (str, optional): If not None, return only datasets of type
            `dataset_type`. Defaults to None.
        dataset_name_contains (str, optional): If not None, return only datasets whose
            name contains `dataset_name_contains`. Defaults to None.
        flm_sess (flm.SESSION, optional): Flexilims session. Defaults to FLM_SESS.

    Returns:
        list: List of flm.schema.Dataset objects
    """
    all_datasets = []
    for rec in recordings:
        datasets = flm.get_children(
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
        all_datasets.append(datasets)
    return all_datasets
