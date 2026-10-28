#!/bin/bash
#SBATCH -p main
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 48:00:00
#SBATCH -J nb13
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/nb13_%j.out
# Cross-evaluation: train with / without 1128(b)(4) positives, test on both label sets.
# 6 runs x 5 folds x 2 models (XGB, B+D features). Resubmit to resume if cut off.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm
SMOKE=0 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output 13_crosseval_nob4_run.ipynb notebooks/13_crosseval_nob4.ipynb
