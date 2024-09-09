#!/bin/bash
#SBATCH --job-name=run_kuzu_backend             # Job name
#SBATCH --partition=short                       # Partition to submit to
#SBATCH --ntasks=1                              # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=1                       # Number of CPU cores per task
#SBATCH --mem=128GB                             # Memory per node
#SBATCH --time=4:00:00                          # Maximum runtime
#SBATCH --output=kuzu.log                       # Output log file

# Load Conda environment
source /home/anonym/miniconda3/etc/profile.d/conda.sh
conda activate python3.9-pyg

# Run the Python script
python kuzu_backend.py
