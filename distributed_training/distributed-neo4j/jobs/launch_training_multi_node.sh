#!/bin/bash
#SBATCH --job-name=pyg-distributed
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pyg_%j.log

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate distributed

# Each task must have a unique rank within the world
RANK=NEED_TO_DETERMINE_NODE_RANK_WHICH_IS_EITHER_0_OR_1

# Export environment variables necessary for distributed training
export MASTER_ADDR=NEED_ADDRESS_NAME_HERE
export TP_SOCKET_IFNAME=NEED_INTERFACE_NAME_HERE
export GLOO_SOCKET_IFNAME=NEED_INTERFACE_NAME_HERE

# Ensure that all nodes start at roughly the same time
srun sleep 10

# Run the distributed training script
srun python node_ogb_cpu.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --num_nodes=$SLURM_NNODES \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR
