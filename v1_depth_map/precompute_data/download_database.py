"""
Utility to generate the JSON file for the database.

Only required if the database has been updated.
"""

import json
import flexiznam as flz
import flexilims as flm

print("Connecting to live flexilims to download 'hey2_3d-vision_foodres_20220101'...")
s1 = flz.get_flexilims_session(
    project_id="hey2_3d-vision_foodres_20220101", offline_mode=False
)
db1 = flm.download_database(s1, types=["mouse", "session", "recording", "dataset"])

print("Connecting to live flexilims to download 'colasa_3d-vision_revisions'...")
s2 = flz.get_flexilims_session(
    project_id="colasa_3d-vision_revisions", offline_mode=False
)
db2 = flm.download_database(s2, types=["mouse", "session", "recording", "dataset"])

print("Merging database snapshots...")
db_data = {**db1, **db2}

target_project = "v1_depth_map"
target_folder = flz.get_data_root(which="processed", project=target_project)
assert target_folder.exists(), f"Target folder {target_folder} does not exist."

db_file = target_folder / "offline_database.json"
print(f"Saving merged database to {db_file}...")
with open(db_file, "w") as f:
    json.dump(db_data, f)

print("Database generation completed successfully!")
