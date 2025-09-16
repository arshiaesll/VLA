#!/bin/sh
#SBATCH --job-name=hybridModels
#SBATCH -N 1    ## requests on 1 node
#SBATCH --gres=gpu:2                 # request 2 GPUs
#SBATCH --output /work/aeslami/VLA_results/job%j.out
#SBATCH --error /work/aeslami/VLA_results/job%j.err
#SBATCH -p gpu-v100-32gb

## Limit threads to be equal to slurm request
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load CUDA module
module load cuda/12.1

# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/2023.9


# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/aeslami/VLA/VLA_env

# Add this line to pass the no_quant flag if it's set to true
accelerate launch --multi-gpu /work/aeslami/VLA/main.py train


