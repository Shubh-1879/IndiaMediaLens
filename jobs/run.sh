#!/bin/bash
#PBS -N Mistral_FineTune_Full
#PBS -l ncpus=32
#PBS -l walltime=12:00:00
#PBS -o out_train.log
#PBS -e err_train.log
#PBS -q gpu
#PBS -M shubham.agarwal_phd24@ashoka.edu.in

module load compiler/anaconda3
cd $PBS_O_WORKDIR

source activate mistral_finetune

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

nvidia-smi dmon -s um -d 10 > gpu_usage.log &

ENV_PYTHON="/home/shubham.agarwal_phd24/.conda/envs/mistral_finetune/bin/python"

$ENV_PYTHON train.py
