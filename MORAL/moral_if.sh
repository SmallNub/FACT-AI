#!/bin/bash
#SBATCH --job-name=moral_train
#SBATCH --output=slurm/moral_train_%j.log
#SBATCH --error=slurm/moral_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=9
#SBATCH --ntasks=1

module purge
module load 2024
module load Anaconda3/2024.06-1

source activate FACT

set -euo pipefail

srun python main.py --fair_model moral --model gae --dataset credit --device cuda:0 --epochs 500 --lr 0.0003 --track_emissions --emissions_dir "emissions_MORAL/" --results_dir "results_MORAL/" --individual
srun python main.py --fair_model moral --model gae --dataset german --device cuda:0 --epochs 500 --lr 0.0003 --track_emissions --emissions_dir "emissions_MORAL/" --results_dir "results_MORAL/" --individual
srun python main.py --fair_model moral --model gae --dataset nba --device cuda:0 --epochs 500 --lr 0.0003 --track_emissions --emissions_dir "emissions_MORAL/" --results_dir "results_MORAL/" --individual
srun python main.py --fair_model moral --model gae --dataset facebook --device cuda:0 --epochs 500 --lr 0.0003 --track_emissions --emissions_dir "emissions_MORAL/" --results_dir "results_MORAL/" --individual
srun python main.py --fair_model moral --model gae --dataset pokec_n --device cuda:0 --epochs 500 --lr 0.0003 --track_emissions --emissions_dir "emissions_MORAL/" --results_dir "results_MORAL/" --individual
srun python main.py --fair_model moral --model gae --dataset pokec_z --device cuda:0 --epochs 500 --lr 0.0003 --track_emissions --emissions_dir "emissions_MORAL/" --results_dir "results_MORAL/" --individual
