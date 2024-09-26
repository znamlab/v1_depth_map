import flexiznam as flz
import yaml
from flexilims.offline import download_database
from tifffile import TiffFile
import numpy as np


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
    if (data_root / recording["path"] / sorted(dataset["tif_files"])[0]).exists():
        tif_path = data_root / recording["path"] / sorted(dataset["tif_files"])[0]
        metadata = TiffFile(tif_path).scanimage_metadata
    else:
        suite2p_ds = flz.get_datasets(
            flexilims_session=flexilims_session,
            origin_name=recording.name,
            dataset_type="suite2p_traces",
            filter_datasets={"anatomical_only": 3},
            allow_multiple=False,
            return_dataseries=False,
        )
        metadata_path = suite2p_ds.path_full / "si_metadata.npy"
        metadata = np.load(metadata_path, allow_pickle=True).item()
    return metadata
