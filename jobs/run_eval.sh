#!/bin/bash
#PBS -N Mistral_Eval
#PBS -l ncpus=16
#PBS -l walltime=01:00:00
#PBS -o out_eval.log
#PBS -e err_eval.log
#PBS -q gpu
#PBS -M shubham.agarwal_phd24@ashoka.edu.in

module load compiler/anaconda3
cd $PBS_O_WORKDIR
source activate mistral_finetune

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Force Hugging Face to operate entirely offline using cached models
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Initiating Mistral ABSA Evaluation..."

# Launch the evaluation script
python eval.py
