#!/bin/bash
#SBATCH --job-name=pyg-distributed-single-node
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pyg-distributed-single-node.log

# Load Conda environment
source /home/anonym/miniconda3/etc/profile.d/conda.sh
conda activate distributed

export MASTER_ADDR=127.0.0.1
export TP_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# Node 0:
nohup python in-memory-alternative.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --num_nodes=2 \
  --node_rank=0 \
  --num_loader_threads=48 \
  --num_workers=32 \
  --concurrency=32 \
  --master_addr=$MASTER_ADDR > logs/single_node_0.log 2>&1 &

# Node 1:
python in-memory-alternative.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --num_nodes=2 \
  --node_rank=1 \
  --num_loader_threads=48 \
  --num_workers=32 \
  --concurrency=32 \
  --master_addr=$MASTER_ADDR > logs/single_node_1.log 2>&1