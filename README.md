# 3d-vision-analysis-2p
Analysis code for 2p imaging for 3d vision project

## Installation

## Step1: Preprocess 2p data
1. Add data to flexilim. Remember to check if the path is corrected uploaded as os path (not windows path).
2. Run `run_suite2p_gpu.sh` from `2p-preprocess` package.
3. Check preprocessed data:
    1. Check registration (no z drift?)
    2. Curate neuron / non-neuron
    3. Update classifier 
4. Synchronise param loggers. 

## Step2: Basic visualization.

## Step3: Model fitting  
1. To use GPU on CAMP:      
    1. Activate the analysis environment, install the jax wheel for CUDA11-CUDNN8.0.8: `pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.html`
    2. In bash script, use gpu node 038 `#SBATCH --nodelist=gpu038`   
    and add:   
    ```
    ml CUDA/11.1.1-GCC-10.2.0
    ml cuDNN/8.0.5.39-CUDA-11.1.1
    ```
    
  
