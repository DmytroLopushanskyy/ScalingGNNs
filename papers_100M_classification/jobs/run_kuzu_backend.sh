#!/bin/bash
#SBATCH --job-name=run_kuzu_backend   # Job name
#SBATCH --partition=short            # Partition to submit to
#SBATCH --ntasks=1                  # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --mem=128GB                  # Memory per node
#SBATCH --time=4:00:00             # Maximum runtime
#SBATCH --output=final_experiments/13.log          # Output log file

# 256 GB times out for 12,12,12
# 384 GB - impossible to provision
# trying 256 for 12 -- NaN exception
# fixed it now, relaunching 256 for 12,12 - job logs stored in run_kuzu_original_output_256_2 (8 CPUs, 256 GB, 4h limit)
# Something is wrong with loss, its nan all the time, need to fix. adding prints and relaunching it

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate py3.9-pyg

# Run the Python script
python kuzu_backend.py
