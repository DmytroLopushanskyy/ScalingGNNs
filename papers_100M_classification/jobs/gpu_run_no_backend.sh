#!/bin/bash
#SBATCH --job-name=run_no_2             # Job name
#SBATCH --partition=short             # Partition to submit to
#SBATCH --ntasks=1                    # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=48            # Number of CPU cores per task
#SBATCH --mem=256GB                   # Memory per node
#SBATCH --gres=gpu:a100:1             # GPU Usage
#SBATCH --time=4:00:00                # Maximum runtime
#SBATCH --output=logs/gpu_run_no_output_2.log  # Output log file
#SBATCH --error=logs/gpu_run_no_error_.log    # Error log file

# 384 GB mem
# 96 CPUs

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate py3.9-pyg

# Run the Python script
python no_backend.py