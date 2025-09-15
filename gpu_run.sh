#!/bin/sh
#SBATCH --job-name=hybridModels
#SBATCH -N 1    ## requests on 1 node
#SBATCH -n 28   ## requests on 1 CPU
#SBATCH --gres=gpu:1 ## requests on 1 GPU
#SBATCH --output /work/aeslami/VLA_results/job%j.out
#SBATCH --error /work/aeslami/VLA_results/job%j.err
#SBATCH -p gpu-v100-16gb,gpu-v100-32gb,gpu,defq-48core,defq-64core

## Limit threads to be equal to slurm request
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load CUDA module, 10.1 worked ok but the quantization gave an error
module load cuda/10.1

# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/2023.9


# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/aeslami/LM_finetune/finetune3/finetune3_env

# Add this line to pass the no_quant flag if it's set to true
python /work/aeslami/LM_finetune/finetune3/main.py
