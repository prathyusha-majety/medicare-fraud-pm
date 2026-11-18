#!/bin/bash
#SBATCH -J xgb_prior
#SBATCH -p main
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/%x_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm
python -u src/xgb_prior_threshold.py