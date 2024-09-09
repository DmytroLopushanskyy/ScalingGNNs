# **Distributed GNN Training with Neo4j Backend**

This directory demonstrates a scalable method for distributed Graph Neural Network (GNN) training by directly integrating Neo4j graph databases. Unlike traditional GNN models that require loading the entire graph into memory, this approach retrieves graph data on demand using Neo4j, leading to significant memory savings and scalability improvements.

## **Repository Structure**
- `main-neo4j-backend.py`: Main script to run the Neo4j-based GNN training.
- `in-memory-alternative.py`: Alternative in-memory distributed training that we compare our method to.
- `partition_graph.py`: For graph partitioning.
- **`jobs/`**: Scripts for launching training jobs.
  - `launch_training_single_node.sh`: Script to launch training on a single node.
- **`loaders/`**: Functions to load data into Neo4j.
  - `load_into_neo4j.py`: Script for loading data into the Neo4j database.
- **`logs/`**: Stores logs from experiments.
  - Example logs: `experiment-256GB-RAM-48CPUs-node0.log`, etc.
- **`plots/`**: Visualised performance metrics from logs.
  - Example plot: `experiment-256GB-RAM-48CPUs.png`.
- **`src/`**: Source code files.
  - `graph_sampler.py`: Main class that implements neighbour sampling. 
  - `client.py`: Interface for interacting with the Neo4j database.
  - `feature_store.py`: Manages feature storage and retrieval.
  - `graph_store.py`: Handles graph storage and access.
- **`README.md`**: Documentation file (this file).
- **`visualize.py`**: Script to generate plots from the logs.

## **System Requirements**
- **Neo4j**: A graph database to manage and query graph data in real-time.
- **PyTorch Geometric (PyG)**: For implementing GNN layers and managing the training process.
- **Docker** (optional but recommended): To maintain consistent environments across nodes.

## **Create Conda Environment**

Before running any training scripts, ensure you have set up a dedicated environment with the necessary libraries. We recommend using **Conda** for environment management.

To set up the environment:
```bash
conda create -n neo4j-backend python=3.9
conda activate neo4j-backend
pip install -r requirements.txt
```

Next, replace the `neighbor_sampler.py` file in the PyTorch Geometric installation:
```bash
cp ScalingGNN/torch_geom_neighbor_sampler.py <miniconda_path>/envs/neo4j-backend/lib/python3.9/site-packages/torch_geometric/sampler/neighbor_sampler.py
```
This modified `neighbor_sampler.py` is critical for running GNN training with Neo4j integration.

For other backends (Kuzu or in-memory), create a duplicate environment without the Neo4j-specific changes:
```bash
conda create --clone neo4j-backend --name kuzu-backend
# Activate the environment
conda activate kuzu-backend
``

## **Installation**

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```

2. Install the necessary dependencies using Conda or your environment directly:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Neo4j:
   - Load the graph dataset into Neo4j using the provided script:
     ```bash
     python loaders/load_into_neo4j.py --dataset ogbn-products
     ```

## **Usage**

### **Running the Distributed Training**

1. **Single Node Execution:**
   To execute training on a single node:
   ```bash
   bash distributed_training/launch_training_single_node.sh
   ```

2. **Distributed Execution:**
   On each node, run the following:
    ```
    conda activate neo4j-backend
    cd /home/anonym/scaling-gnns/distributed-neo4j
    export MASTER_ADDR=127.0.0.1
    export TP_SOCKET_IFNAME=lo
    export GLOO_SOCKET_IFNAME=lo
    ```

   Then proceed:
    ```bash
    # Node 0:
    python -u main-neo4j-backend.py \
      --dataset=ogbn-products \
      --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
      --master_addr=$MASTER_ADDR \
      --num_loader_threads=8 \
      --batch_size=1024 \
      --num_workers=0 \
      --num_nodes=2 \
      --node_rank=0

    # Node 1:
    python -u main-neo4j-backend.py \
      --dataset=ogbn-products \
      --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
      --master_addr=$MASTER_ADDR \
      --num_loader_threads=8 \
      --batch_size=1024 \
      --num_workers=0 \
      --num_nodes=2 \
      --node_rank=1
    ```

### **Alternative: In-Memory Training Approach**

To run the alternative in-memory GNN training method:
```bash
# Node 0:
python in-memory-alternative.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --master_addr=$MASTER_ADDR \
  --concurrency=4 \
  --num_workers=2 \
  --num_loader_threads=8 \
  --batch_size=8192 \
  --num_nodes=2 \
  --node_rank=0

# Node 1:
python in-memory-alternative.py \
  --dataset=ogbn-products \
  --dataset_root_dir=./data/partitions/ogbn-products/2-parts \
  --master_addr=$MASTER_ADDR \
  --concurrency=16 \
  --num_workers=16 \
  --num_loader_threads=8 \
  --batch_size=1024 \
  --num_nodes=2 \
  --node_rank=1
```

## **Data Partitioning**

Before running the in-memory approach, partition the dataset:
```bash
python src/partition_graph.py --dataset=ogbn-products --num_partitions=2
```

The partitioned data will be stored under the `data/partitions` folder.

## **Logs and Visualisation**

- Training logs will be stored under the `logs/` folder.
- Use `visualize.py` to generate performance plots:
   ```bash
   python visualize.py --log_dir=logs/
   ```

## **Acknowledgements**

This project builds upon PyTorch Geometric's distributed framework and Neo4j for handling large-scale graphs.