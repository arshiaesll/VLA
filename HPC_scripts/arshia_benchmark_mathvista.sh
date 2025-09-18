#!/bin/sh
#SBATCH --job-name=VLA
#SBATCH -N 1    ## requests on 1 node
#SBATCH --gres=gpu:1                 # request 2 GPUs
#SBATCH --output /work/aeslami/VLA_results/job%j.out
#SBATCH --error /work/aeslami/VLA_results/job%j.err
#SBATCH -p gpu-H200

## Limit threads to be equal to slurm request
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Load CUDA module
module load cuda12.4

# Load Anaconda module (adjust the path if necessary)
module load python3/anaconda/3.12


# Activate your conda environment, ## user source activate on cluster, not conda activate
source activate /work/aeslami/VLA/VLA_env

# Add this line to pass the no_quant flag if it's set to true
python /work/aeslami/VLA/main.py benchmark --from-finetuned /work/aeslami/VLA/Models/microsoft/phi-3.5-vision-instruct-microsoft/phi-3.5-vision-instruct/checkpoint-5000


