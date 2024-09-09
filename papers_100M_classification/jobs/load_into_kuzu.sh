#!/bin/bash
#SBATCH --job-name=load_into_kuzu   # Job name
#SBATCH --partition=long            # Partition to submit to
#SBATCH --ntasks=1                  # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=10           # Number of CPU cores per task
#SBATCH --mem=128GB                  # Memory per node
#SBATCH --time=24:00:00             # Maximum runtime
#SBATCH --output=load_into_kuzu_output.log          # Output log file
#SBATCH --error=load_into_kuzu_error.log            # Error log file

# Load Conda environment
source /home/anonym/miniconda3/etc/profile.d/conda.sh
conda activate python3.9-pyg

# Run the Python script
python ../loaders/load_into_kuzu.py
