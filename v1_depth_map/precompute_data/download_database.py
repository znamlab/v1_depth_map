"""
Utility to generate the JSON file for the database.

Only required if the database has been updated.
"""

import json
import flexiznam as flz
import flexilims as flm

project = "hey2_3d-vision_foodres_20220101"
flm_sess = flz.get_flexilims_session(
    project_id="hey2_3d-vision_foodres_20220101", offline_mode=False
)

db_data = flm.download_database(
    flm_sess, types=["mouse", "session", "recording", "dataset"]
)

target_project = "v1_depth_map"
target_folder = flz.get_data_root(which="processed", project=target_project)
assert target_folder.exists(), f"Target folder {target_folder} does not exist."

db_file = target_folder / "offline_database.json"
with open(db_file, "w") as f:
    json.dump(db_data, f)
