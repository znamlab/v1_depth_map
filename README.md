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
  
