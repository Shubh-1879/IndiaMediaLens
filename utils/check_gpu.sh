#!/bin/bash
#PBS -N GPU_Check
#PBS -l nodes=1
#PBS -q gpu

#!/bin/bash
#PBS -N GPU_Check
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q gpu

module load compiler/anaconda3

# Put the system CUDA 11.7 FIRST to avoid Anaconda duplicates
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export BNB_CUDA_VERSION=118
export CUDA_HOME=/usr/local/cuda-11.7

cd $PBS_O_WORKDIR
source activate mistral_finetune

# Run the diagnostic
python -m bitsandbytes > bnb_report_final.txt 2>&1
