#!/bin/bash
#SBATCH --job-name=run_kuzu_backend   # Job name
#SBATCH --partition=long            # Partition to submit to
#SBATCH --ntasks=1                  # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=10           # Number of CPU cores per task
#SBATCH --mem=370GB                  # Memory per node
#SBATCH --time=24:00:00             # Maximum runtime
#SBATCH --output=logs/run_kuzu_backend_output.log          # Output log file
#SBATCH --error=logs/run_kuzu_backend_error.log            # Error log file

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate py3.9-pyg

# Run the Python script
python kuzu_backend.py
