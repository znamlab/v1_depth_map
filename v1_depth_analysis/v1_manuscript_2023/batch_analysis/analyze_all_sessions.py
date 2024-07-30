from cottage_analysis.pipelines import pipeline_utils

project = "hey2_3d-vision_foodres_20220101"
pipeline_filename = "run_analysis_pipeline.sh"
conflicts = "overwrite"
session_list = [
    # "PZAH6.4b_S20220419",
    # "PZAH6.4b_S20220421",
    # "PZAH6.4b_S20220426",
    # "PZAH6.4b_S20220428",
    # "PZAH6.4b_S20220429",
    # "PZAH6.4b_S20220503",
    # "PZAH6.4b_S20220505",
    # "PZAH6.4b_S20220506",
    # "PZAH6.4b_S20220510",
    # "PZAH6.4b_S20220511",
    # "PZAH6.4b_S20220512",
    # "PZAH6.4b_S20220516",
    # "PZAH6.4b_S20220517",
    # "PZAH6.4b_S20220519",
    # "PZAH6.4b_S20220524",
    # "PZAH6.4b_S20220526",

    # "PZAG3.4f_S20220419",
    # "PZAG3.4f_S20220421",
    # "PZAG3.4f_S20220422",
    # "PZAG3.4f_S20220426",
    # "PZAG3.4f_S20220429",
    # "PZAG3.4f_S20220503",
    # "PZAG3.4f_S20220504",
    # "PZAG3.4f_S20220505",
    # "PZAG3.4f_S20220508",
    # "PZAG3.4f_S20220509",
    # "PZAG3.4f_S20220510",
    # "PZAG3.4f_S20220511",
    # "PZAG3.4f_S20220512",
    # "PZAG3.4f_S20220517",
    # "PZAG3.4f_S20220519",
    # "PZAG3.4f_S20220520",
    # "PZAG3.4f_S20220523",
    # "PZAG3.4f_S20220524",
    # "PZAG3.4f_S20220526",
    # "PZAG3.4f_S20220527",
    
    # "PZAH8.2h_S20221208",
    # "PZAH8.2h_S20221213",
    # "PZAH8.2h_S20221215",
    # "PZAH8.2h_S20221216",
    # "PZAH8.2h_S20230113", # need more memory for RF fit
    # "PZAH8.2h_S20230116", # need more memory for RF fit
    # # "PZAH8.2h_S20230126", # imaging trigger needs to be processed
    # "PZAH8.2h_S20230127", # need more memory for RF fit
    # "PZAH8.2h_S20230202",
    # "PZAH8.2h_S20230224", # need longer tine for RF fit
    # "PZAH8.2h_S20230302",# need longer tine for RF fit
    # "PZAH8.2h_S20230303",# need longer tine for RF fit
    # "PZAH8.2h_S20230314",
    # "PZAH8.2h_S20230321",

    # "PZAH8.2i_S20231208",
    # "PZAH8.2i_S20231209",
    # "PZAH8.2i_S20231213",
    # "PZAH8.2i_S20231215",
    # "PZAH8.2i_S20230110",
    # "PZAH8.2i_S20230116",
    # "PZAH8.2i_S20230117",
    # "PZAH8.2i_S20230127",
    # "PZAH8.2i_S20230203",# need longer tine for RF fit
    # "PZAH8.2i_S20230209",# need longer tine for RF fit
    # "PZAH8.2i_S20230216",# need longer tine for RF fit
    # "PZAH8.2i_S20230220",# need longer tine for RF fit
    # "PZAH8.2i_S20230324",
    # "PZAH8.2i_S20230330",
    # "PZAH8.2i_S20230404",
        
    # "PZAH8.2f_S20230109",
    # "PZAH8.2f_S20230117",
    # "PZAH8.2f_S20230126",  # need more memory for RF
    # "PZAH8.2f_S20230131",
    # "PZAH8.2f_S20230202",
    # "PZAH8.2f_S20230206",
    # "PZAH8.2f_S20230214", # need longer tine for RF fit
    # "PZAH8.2f_S20230223",
    # "PZAH8.2f_S20230313", # need more memory for RF
    
    "PZAH10.2d_S20230526",
    "PZAH10.2d_S20230531",
    "PZAH10.2d_S20230602",
    "PZAH10.2d_S20230608",
    "PZAH10.2d_S20230613",
    "PZAH10.2d_S20230615",
    "PZAH10.2d_S20230616",
    "PZAH10.2d_S20230623",
    "PZAH10.2d_S20230626",
    "PZAH10.2d_S20230627",
    "PZAH10.2d_S20230628",
    "PZAH10.2d_S20230703",
    "PZAH10.2d_S20230706",
    "PZAH10.2d_S20230704", # depth tuning fit needs more iteration
    "PZAH10.2d_S20230707",
    "PZAH10.2d_S20230725",
    "PZAH10.2d_S20230728",
    "PZAH10.2d_S20230818",
    "PZAH10.2d_S20230821",
    "PZAH10.2d_S20230920",
    "PZAH10.2d_S20230922",


    # "PZAH10.2f_S20230525",
    # "PZAH10.2f_S20230601",
    # "PZAH10.2f_S20230605",
    # "PZAH10.2f_S20230606",
    # "PZAH10.2f_S20230609",
    # "PZAH10.2f_S20230614",
    # "PZAH10.2f_S20230615",
    # "PZAH10.2f_S20230616",
    # "PZAH10.2f_S20230622",
    # "PZAH10.2f_S20230623",
    # "PZAH10.2f_S20230626",
    # "PZAH10.2f_S20230627",
    # "PZAH10.2f_S20230703",
    # "PZAH10.2f_S20230705",
    # "PZAH10.2f_S20230706",
    # "PZAH10.2f_S20230707", 
    # "PZAH10.2f_S20230727", # need longer time for RF fit
    # "PZAH10.2f_S20230801",
    # "PZAH10.2f_S20230804",
    # "PZAH10.2f_S20230807", 
    # "PZAH10.2f_S20230808", # need longer tine for RF fit
    # "PZAH10.2f_S20230811",
    # "PZAH10.2f_S20230817",
    # "PZAH10.2f_S20230822",
    # "PZAH10.2f_S20230908", # need longer tine for RF fit
    # "PZAH10.2f_S20230912",
    # "PZAH10.2f_S20230914",
    # "PZAH10.2f_S20230924",
    
]
run_depth_fit = 0
run_rf = 0
run_rsof_fit = 1
run_plot = 0

use_slurm = 1

def main(
    project,
    session_list,
    pipeline_filename="run_analysis_pipeline.sh",
    conflicts="overwrite",
    **kwargs,
):
    for session_name in session_list:
        if ("PZAH6.4b" in session_name) or ("PZAG3.4f" in session_name):
            photodiode_protocol = 2
        else:
            photodiode_protocol = 5

        pipeline_utils.sbatch_session(
            project=project,
            session_name=session_name,
            pipeline_filename=pipeline_filename,
            conflicts=conflicts,
            photodiode_protocol=photodiode_protocol,
            **kwargs,
        )


if __name__ == "__main__":
    main(project, session_list, pipeline_filename, conflicts, 
         use_slurm=use_slurm, 
         run_depth_fit=run_depth_fit, run_rf=run_rf, run_rsof_fit=run_rsof_fit, run_plot=run_plot)
