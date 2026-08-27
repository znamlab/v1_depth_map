from cottage_analysis.pipelines import pipeline_utils

project = "hey2_3d-vision_foodres_20220101"
pipeline_filename = "run_analysis_pipeline_size_control.sh"
conflicts = "overwrite"
session_list = [
    "PZAH10.2d_S20230822",
    "PZAH10.2f_S20230815",
    "PZAH10.2f_S20230907",
]

# PZAH10.2f_S20230815's three recordings are *named* SpheresPermTubeReward but hold
# genuine size-control data (OriginalSize sweeps [0.0435, 0.087, 0.174]; plain closed-loop
# recordings hold a single 0.087). That misleading name needs no override here:
# size_control.sync_all_recordings() filters on the `protocol` *attribute*
# (`recordings.protocol == protocol_base`), not the recording name, and flexilims records
# protocol='SizeControl' for all three. Overriding to "SpheresPermTubeReward" matches zero
# recordings and fails with an UnboundLocalError on vs_df_all.
PROTOCOL_BASE_OVERRIDES = {}
DEFAULT_PROTOCOL_BASE = "SizeControl"

use_slurm = 0
log_fname = "size"


def main(
    project,
    session_list,
    pipeline_filename="run_analysis_pipeline.sh",
    conflicts="overwrite",
    use_slurm=False,
    **kwargs,
):
    for session_name in session_list:
        if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5

        protocol_base = PROTOCOL_BASE_OVERRIDES.get(session_name, DEFAULT_PROTOCOL_BASE)
        print(f"{session_name}: protocol_base={protocol_base}")

        pipeline_utils.sbatch_session(
            project=project,
            session_name=session_name,
            pipeline_filename=pipeline_filename,
            conflicts=conflicts,
            photodiode_protocol=photodiode_protocol,
            use_slurm=use_slurm,
            protocol_base=protocol_base,
            **kwargs,
        )


if __name__ == "__main__":
    main(
        project,
        session_list,
        pipeline_filename,
        conflicts,
        use_slurm,
        log_fname=log_fname,
        use_annotated=False,
        ast_neuropil=False,
    )
