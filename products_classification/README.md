# **Single-Machine Training with Neo4j and In-Memory for Products Classification**

This repository demonstrates single-machine GNN training for node classification on the **Products** dataset, using two backends: Neo4j and in-memory processing. The approach integrates Neo4j graph databases to manage graph data and supports memory-efficient training by relying on external graph stores.

## **Overview**

The **Products dataset** is part of the Open Graph Benchmark (OGB) and is designed for node classification tasks. Each node represents a product, and each edge signifies a co-purchase event between products. The goal is to predict the product category based on its connections to other products.

We demonstrate GNN training using two backends:
1. **Neo4j**: Graph data is queried in real-time from a Neo4j database, allowing efficient memory use.
2. **In-Memory**: The entire graph is loaded into memory for traditional training.

### **Setup and Installation**

#### **1. Create Conda Environment**

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
This modified `neighbor_sampler.py` is crucial for running GNN training with Neo4j integration.

For the in-memory method, create a duplicate environment without the Neo4j-specific changes:
```bash
conda create --clone neo4j-backend --name no-backend
# Activate the environment
conda activate no-backend
```

### **Running the Models**

#### **Neo4j Backend**

Ensure Neo4j is running, and the Products dataset is loaded into the database. You can refer to the **distributed_training** folder for guidance on how to set up Neo4j for distributed training. Once the Neo4j instance is ready, run the training with the Neo4j backend using:
```bash
python neo4j_backend.py
```

#### **In-Memory**

For the traditional in-memory method, simply run:
```bash
python no_backend.py
```

### **Repository Structure**

- **`neo4j_backend.py`**: Script for Neo4j backend operations.
- **`no_backend.py`**: Script for in-memory operations.

- **`plots/`**: Contains visualisations of memory usage and other performance metrics.
  - `memory_usage_plot.png`: Plot showing memory usage during training.

- **`src/`**: Core code for training and backend interaction.
  - `client.py`: Handles the connection and querying from Neo4j.
  - `feature_store.py`: Manages feature storage and retrieval.
  - `graph_sage.py`: GraphSAGE model architecture for node classification.
  - `graph_sampler.py`: Custom graph sampling for efficient training.
  - `graph_store.py`: Manages graph data retrieval from Neo4j.
  - `train_test.py`: Auxiliary script to handle training logic (not called directly).


## **References**

For more information on the methods and results, please refer to the report on scaling GNNs with graph databases.