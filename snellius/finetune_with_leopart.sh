#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=18
#SBATCH --gpus=2
#SBATCH --partition=gpu
#SBATCH --time=60:00:00
#SBATCH --exclude=gcn45,gcn59
#SBATCH --output=finetune_with_leopart.out
#SBATCH --job-name=finetune_with_leopart

# Execute program located in $HOME

source activate obdet2

srun python experiments/finetune_with_leopart.py
