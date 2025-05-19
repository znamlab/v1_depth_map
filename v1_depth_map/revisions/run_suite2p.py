# Cell to re-run suite2p

# If you want to run this cell, you'll need extra dependencies of 2p-preprocess:
# pip install suite2p, jax, optax --no-deps
# And twop_preprocess itself
# Note that if use_slurm is true, they just need to be imported successfully, another job
# will actually run the code
import flexiznam as flz
from pathlib import Path
from twop_preprocess.calcium import extract_session


project = "colasa_3d-vision_revisions"
conflicts = "overwrite"
use_slurm=False
run_split = True
run_suite2p = False
run_dff = True
delete_previous_run= False
replace_is_cells = False

sessions = {
    "PZAG17.3a_S20250402": "motor",
    "PZAG17.3a_S20250319": "multidepth",
    "PZAG17.3a_S20250306": "spheretube_5",
    "PZAG17.3a_S20250305": "spheretube_4",
    "PZAG17.3a_S20250303": "spheretube_3",
    "PZAG17.3a_S20250228": "spheretube_2",
    "PZAG17.3a_S20250227": "spheretube_1",
    "PZAG16.3c_S20250401": "motor",
    "PZAG16.3c_S20250317": "multidepth",
    "PZAG16.3c_S20250313": "spheretube_5",
    "PZAG16.3c_S20250310": "spheretube_4",
    "PZAG16.3c_S20250221": "spheretube_3",
    "PZAG16.3c_S20250220": "spheretube_2",
    "PZAG16.3c_S20250219": "spheretube_1",
    "PZAG16.3b_S20250401": "motor",
    "PZAG16.3b_S20250317": "multidepth",
    "PZAG16.3b_S20250313": "spheretube_5",
    "PZAG16.3b_S20250310": "spheretube_4",
    "PZAG16.3b_S20250226": "spheretube_3",
    "PZAG16.3b_S20250225": "spheretube_2",
    "PZAG16.3b_S20250224": "spheretube_1",
    "PZAH17.1e_S20250403": "motor",
    "PZAH17.1e_S20250318": "multidepth",
    "PZAH17.1e_S20250313": "multidepth",
    "PZAH17.1e_S20250311": "spheretube_5",
    "PZAH17.1e_S20250307": "spheretube_4",
    "PZAH17.1e_S20250306": "spheretube_3",
    "PZAH17.1e_S20250305": "spheretube_2",
    "PZAH17.1e_S20250304": "spheretube_1",
}


print(f"{len(sessions)} sessions to analyze")


slurm_folder = Path.home() / "slurm_logs" / project
slurm_folder.mkdir(exist_ok=True, parents=True)
flm_sess = flz.get_flexilims_session(project_id=project)
for session_name in sessions:
    if session_name != "PZAH17.1e_S20250403":
        continue
    mouse_name = session_name.split("_")[0]
    mouse = flz.get_entity(
        name=mouse_name, datatype="mouse", flexilims_session=flm_sess
    )
    gcamp = mouse.genotype_text.split("GCaMP")[1][:2]
    if gcamp == "6f":
        flow_threshold = 2
        cellprob_threshold = 0
    elif gcamp == "6s":
        flow_threshold = 2
        cellprob_threshold = -2
    else:
        raise ValueError(
            f"Unknown Gcamp version: {gcamp} in mouse {mouse_name} with"
            + f" genotype: {mouse.genotype_text}"
        )
    ops = {
        "tau": 0.7,
        "ast_neuropil": False,
        "delete_bin": False,
        "move_bin": True,
        "roidetect": True,
        "flow_threshold": flow_threshold,
        "cellprob_threshold": cellprob_threshold,
        "diameter_multiplier": 0.015,
    }
    # delete None values
    ops = {k: v for k, v in ops.items() if v is not None}
    if not any([run_suite2p, run_dff, run_split]):
        print('Nothing to do with extraction')
    else:
        extract_session(
            project,
            session_name,
            conflicts=conflicts,
            run_split=run_split,
            run_suite2p=run_suite2p,
            run_dff=run_dff,
            ops=ops,
            use_slurm=use_slurm,
            slurm_folder=slurm_folder,
            scripts_name=f"extract_{session_name}",
            delete_previous_run=delete_previous_run,
            
        )
    if replace_is_cells:
        raise NotImplementedError()