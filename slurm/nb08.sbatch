#!/bin/bash
#SBATCH -p main
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 48:00:00
#SBATCH -J nb08
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/nb08_%j.out
# Full Part D baseline (30 folds, XGB + RF), once per label column.
# Results resume from the fold CSV if the job is cut off: just resubmit.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm
for LABEL in fraud fraud_h2; do
  echo "=== label: $LABEL"
  SMOKE=0 LABEL_COL=$LABEL jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 \
    --output 08_baseline_partd_${LABEL}_run.ipynb notebooks/08_baseline_partd.ipynb
done
