#!/bin/bash
#SBATCH --job-name=moral_train
#SBATCH --output=moral_train_%j.log
#SBATCH --error=moral_train_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks=1

set -euo pipefail

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate FACT

srun python main.py --fair_model moral --model gae --dataset credit --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset german --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset nba --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset facebook --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset pokec_n --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset pokec_z --device cuda:0 --epochs 500
