#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --output=slurm/generate_data_%j.log
#SBATCH --error=slurm/generate_data_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=9
#SBATCH --ntasks=1

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate FACT

srun python generate_data.py --all