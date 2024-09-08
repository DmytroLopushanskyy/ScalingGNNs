#!/bin/bash
#SBATCH --job-name=no_backend          # Job name
#SBATCH --partition=short              # Partition to submit to
#SBATCH --ntasks=1                    # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=256GB                   # Memory per node
#SBATCH --time=04:00:00               # Maximum runtime
#SBATCH --output=final_experiments/15.log  # Output log file

# 150 GB - OOM
# Set params to neighbors:[12,12,12], batch_size:1152, num_workers:4
# Trying with 256 Gb, 48 CPUs, medium, 12h, ARC -- took 50 mins to provision
# Trying with 256 Gb, 48 CPUs, short, 12h, ARC -- took 50 mins to provision
# For all of these, loss is being computed correctly but it does not get further than a 2 epochs after 12 hours. I changed accuracy testing to be done after each epoch. Restarting 
# Let's relaunch it with GPU (any) on HTC (short and medium), same params as before ([12,12,12],1152,4 + 256 Gb, 48 CPUs, short, 12h)

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate py3.9-pyg

# Run the Python script
python no_backend.py