#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --output=pretrain_dino_s2c.out
#SBATCH --job-name=pretrain_dino_s2c

# Execute program located in $HOME

source activate obdet

srun python src/benchmark/pretrain_ssl/pretrain_dino_s2c_original.py
