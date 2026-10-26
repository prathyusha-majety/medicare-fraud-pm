#!/bin/bash
#SBATCH -J check_data
#SBATCH -p main
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/%x_%j.out

source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm/src
python check_data.py
