# **Single-Machine Training with Neo4j, Kuzu, and In-Memory for Cora Classification**

This repository demonstrates single-machine GNN training for node classification on the Cora dataset, using different backends: Neo4j, Kuzu, and in-memory processing. The approach integrates Neo4j and Kuzu graph databases to manage graph data and supports memory-efficient training by relying on these external graph stores.

## **Overview**

The **Cora dataset** is widely used for node classification tasks in the context of GNN research. It contains 2,708 publications classified into one of seven classes, with 5,429 citation links between them. Each paper is described by a sparse feature vector, and the goal is to classify the papers based on their citation network.

Our task involves classifying each publication (represented as a node) based on its citation links (edges) to other papers.

We demonstrate GNN training using three different backends:
1. **Neo4j**: Graph data is queried in real-time from a Neo4j database, allowing efficient memory use.
2. **Kuzu**: An embeddable graph database that allows for efficient disk-based querying.
3. **In-Memory**: The graph is fully loaded into memory for traditional training.

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
This modified `neighbor_sampler.py` is critical for running GNN training with Neo4j integration.

For other backends (Kuzu or in-memory), create a duplicate environment without the Neo4j-specific changes:
```bash
conda create --clone neo4j-backend --name kuzu-backend
# Activate the environment
conda activate kuzu-backend
```

### **Running the Model**

#### **Neo4j Backend**

Ensure Neo4j is running and the Cora dataset is loaded into the database. You can use the script provided:
```bash
python loaders/load_cora_into_neo4j.py
```

Then, run the training using the Neo4j backend:
```bash
python src/train_test.py --backend neo4j
```

#### **Kuzu Backend**

To use Kuzu, first load the dataset:
```bash
python loaders/load_cora_into_kuzu_v1.py
```

Run the training with Kuzu:
```bash
python src/train_test.py --backend kuzu
```

#### **In-Memory**

For the traditional in-memory method, simply run:
```bash
python src/train_test.py --backend memory
```

## **Repository Structure**

- **`config/`**: Contains configuration files for different applications.
  - `loader_params.json`: Parameters for loading the dataset.
  - `model_params.json`: Parameters for the model architecture and training.
  
- **`data/`**: Contains the dataset in various formats.
  - `Cora/`: Raw and preprocessed Cora data.
  - `kuzu/`: Kuzu-compatible data storage.

- **`loaders/`**: Scripts to load the Cora dataset into different backends.
  - `load_cora_into_neo4j.py`: Loads Cora into a Neo4j database.
  - `load_cora_into_kuzu_v1.py`: Loads Cora into Kuzu version 1.
  - `query_kuzu.py`: Script for querying Kuzu during training.

- **`src/`**: Core code for training and backend interaction.
  - `client_for_cora.py`: Interfaces with the Cora dataset in Neo4j.
  - `graph_sampler.py`: Custom graph sampling for efficient training.
  - `graph_store_for_cora.py`: Manages graph data retrieval from Neo4j and Kuzu.
  - `train_test.py`: Main training script, supporting different backends.
  - `utils.py`: Utility functions for logging, processing results, etc.
  - `neo4j_backend.py`: Main script for Neo4j backend operations.
  - `kuzu_backend.py`: Main script for Kuzu backend operations.
  - `no_backend.py`: Main script for in-memory operations.
  - `test_backend.py`: Main script to test backend implementations.

## **References**

For more information on the methods and results, please refer to the report on scaling GNNs with graph databases.