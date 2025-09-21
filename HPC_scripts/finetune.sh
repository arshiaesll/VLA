#!/bin/sh
#SBATCH --job-name=VLMFinetune
#SBATCH -N 1    ## requests on 1 node
#SBATCH --gres=gpu:1   # request 2 GPUs
#SBATCH --time=12:00:00
#SBATCH --output /work/aeslami/VLA_results/job%j.out
#SBATCH --error /work/aeslami/VLA_results/job%j.err
#SBATCH -p gpu-H200


# Load CUDA module
module load cuda12.4

# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/3.12


# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/aeslami/VLA/VLA_env

# Add this line to pass all the arguments down to main.py
accelerate launch /work/aeslami/VLA/main.py "$@"

