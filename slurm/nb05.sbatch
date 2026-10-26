   #!/bin/bash
   #SBATCH -p main
   #SBATCH -c 16
   #SBATCH --mem=96G
   #SBATCH -t 48:00:00
   #SBATCH -J nb05
   #SBATCH -o /scratch/sm3664/medicare-fraud/logs/nb05_%j.out
   source ~/miniforge3/etc/profile.d/conda.sh
   conda activate mfraud
   cd ~/medicare-fraud-pm
   SMOKE=0 jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=-1 \
     --output 05_baseline_run.ipynb notebooks/05_baseline.ipynb