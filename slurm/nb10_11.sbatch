#!/bin/bash
#SBATCH -p main
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 72:00:00
#SBATCH -J nb10_11
#SBATCH -o /scratch/sm3664/medicare-fraud/logs/nb10_11_%j.out
# Build the combined Part B + Part D dataset (NB10), then the 30-fold XGB baseline (NB11)
# for each label. Resubmit to resume NB11 if cut off (NB10 is skipped if its output exists).
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mfraud
cd ~/medicare-fraud-pm
if [ ! -f /scratch/sm3664/medicare-fraud/processed/combined/bd_labeled_2019_2023.parquet ]; then
  echo "=== NB10"
  jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output 10_combine_bd_run.ipynb notebooks/10_combine_bd.ipynb || exit 1
fi
for LABEL in fraud fraud_nob4 fraud_h2; do
  echo "=== NB11 label: $LABEL"
  SMOKE=0 LABEL_COL=$LABEL jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output 11_baseline_bd_${LABEL}_run.ipynb notebooks/11_baseline_bd.ipynb
done
