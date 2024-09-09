#!/bin/bash
#SBATCH --job-name=transform_npy_to_csv   # Job name
#SBATCH --partition=long            # Partition to submit to
#SBATCH --ntasks=1                  # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=10           # Number of CPU cores per task
#SBATCH --mem=128GB                  # Memory per node
#SBATCH --time=24:00:00             # Maximum runtime
#SBATCH --output=logs/transform_npy_to_csv_output.log          # Output log file
#SBATCH --error=logs/transform_npy_to_csv_error.log            # Error log file

# Load Conda environment
source /home/anonym/miniconda3/etc/profile.d/conda.sh
conda activate python3.9-pyg

# Run the Python script
python loaders/transform_npy_to_csv.py
