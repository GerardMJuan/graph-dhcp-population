#!/bin/bash
#SBATCH -J vitae
#SBATCH -p high
#SBATCH --mem 32G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o logs_hpc/ae_%j.out
#SBATCH -e logs_hpc/ae_%j.err

export PATH="/home/gmarti/miniconda3/bin:$PATH"
source activate pytorch2

python 3Dautoencoder.py --config configs/vitae_base.yaml