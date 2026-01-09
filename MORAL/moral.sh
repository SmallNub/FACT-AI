#!/bin/bash
#SBATCH --job-name=gpt_train
#SBATCH --output=gpt_train_%j.log
#SBATCH --error=gpt_train_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks=1

set -euo pipefail

srun python main.py --fair_model moral --model gae --dataset credit --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset german --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset nba --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset facebook --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset pokec_n --device cuda:0 --epochs 500
srun python main.py --fair_model moral --model gae --dataset pokec_z --device cuda:0 --epochs 500
