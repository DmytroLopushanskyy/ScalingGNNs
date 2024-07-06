#!/bin/bash
#SBATCH --job-name=pyg-distributed
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=pyg_%j.log

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate py3.9-pyg

# Set MASTER_ADDR to the IP of the first node in the job
MASTER_ADDR=$(hostname -i)

# Set unique port for each job to avoid conflicts
MASTER_PORT=$((12345 + $SLURM_JOB_ID % 10000))

# Total number of tasks across all nodes
WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))

# Each task must have a unique rank within the world
RANK=$(($SLURM_NODEID * $SLURM_NTASKS_PER_NODE + $SLURM_LOCALID))

# Export environment variables necessary for distributed training
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE
export RANK

# Ensure that all nodes start at roughly the same time
srun sleep 10

# Run the distributed training script
srun python node_ogb_cpu.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --num_nodes=$SLURM_NNODES \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR
