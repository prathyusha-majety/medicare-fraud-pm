#!/bin/bash
#SBATCH -p main
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J nb08ae
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/nb08ae_%j.out
# Full Part D baseline on the enriched file: SbP, A, AE x XGB, RF x 30 folds, per label.
# Results go to results/part_d/nb08_AE_<label>_*.csv. Resubmit to resume if cut off.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm
for LABEL in fraud fraud_h2; do
  echo "=== label: $LABEL"
  SMOKE=0 LABEL_COL=$LABEL PD_FILE=part_d_AE_labeled_2019_2023.parquet \
    jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output 08_baseline_partd_AE_${LABEL}_run.ipynb notebooks/08_baseline_partd_v2.ipynb
done
