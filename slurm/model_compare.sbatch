#!/bin/bash
#SBATCH -p main
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/%x_%j.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm
python -u src/model_compare.py "$1"