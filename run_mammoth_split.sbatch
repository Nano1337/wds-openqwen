#!/bin/bash
#SBATCH -J run_mammoth_split
#SBATCH -o logs/run_mammoth_split.out
#SBATCH -e logs/run_mammoth_split.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --mem=512G
#SBATCH --qos=high_priority_nice
#SBATCH --partition=main
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=0
#SBATCH --reservation=haoli_resv


set -euo pipefail

source /fsx/users/haoli/datopenqwen/.venv/bin/activate
python /fsx/users/haoli/datopenqwen/data_prepare/split_mammoth_10m.py
