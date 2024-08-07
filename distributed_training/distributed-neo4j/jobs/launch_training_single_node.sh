#!/bin/bash
#SBATCH --job-name=pyg-distributed-single-node-48_32_32
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pyg-distributed-single-node-48_32_32.log

# First, I started with 1 node, 32G, 10 CPUs, 2 hours max -- this hit the time limit and only processed 5 full train epochs
# Now I start with 1 node, 256 GB, 48 CPUs, 12 hours max -- good!
# Now I start with 1 node, 256 GB, 48 CPUs, 12 hours max but relaunch with 48 threads instead of 1o (to match number of CPUs) 
# same as above plus --num_workers=32 (up from 4)
# same as above plus --concurrency=32 (up from 4)

# Load Conda environment
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate distributed

export MASTER_ADDR=127.0.0.1
export TP_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Node 0:
nohup python node_ogb_cpu.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --num_nodes=2 \
  --node_rank=0 \
  --num_loader_threads=48 \
  --num_workers=32 \
  --concurrency=32 \
  --master_addr=$MASTER_ADDR > logs/single_node_0_48_32_32.log 2>&1 &

# Node 1:
python node_ogb_cpu.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --num_nodes=2 \
  --node_rank=1 \
  --num_loader_threads=48 \
  --num_workers=32 \
  --concurrency=32 \
  --master_addr=$MASTER_ADDR > logs/single_node_1_48_32_32.log 2>&1