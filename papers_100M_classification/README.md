# **Single-Machine Training with Neo4j, Kuzu, and In-Memory for OGBN-Papers100M Classification**

This repository demonstrates single-machine GNN training for node classification on the **ogbn-papers100M** dataset, using three backends: Neo4j, Kuzu, and in-memory processing.

## **Overview**

The **ogbn-papers100M** dataset is part of the Open Graph Benchmark (OGB), consisting of 111 million nodes (papers) and 1.6 billion edges (citations). Each paper is associated with a feature vector. The dataset is ideal for benchmarking GNN models on large-scale node property prediction tasks.

We demonstrate GNN training using three backends:
1. **Neo4j**: Real-time queries from a Neo4j database to manage large-scale graph data.
2. **Kuzu**: A disk-based graph database optimized for handling large datasets like ogbn-papers100M.
3. **In-Memory**: The graph data is fully loaded into memory for traditional training.

### **Setup and Installation**

#### **1. Create Conda Environment**

Before running any training or data manipulation scripts, set up a dedicated environment with the required libraries. We recommend using **Conda** for environment management.

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

For the Kuzu and in-memory methods, create separate environments without the Neo4j-specific modifications:
```bash
conda create --clone neo4j-backend --name kuzu-backend
conda create --clone neo4j-backend --name no-backend
```
Activate the appropriate environment when switching between backends.

### **Data Manipulation and Preparation**

Given the size of the **ogbn-papers100M** dataset, specific scripts are provided for data manipulation and formatting. These scripts help preprocess the dataset into a format that can be efficiently loaded into the backends. 

## Kuzu Loading 

Loading a large csv file into Kuzu requires significant memory usage. For example, a job loading 30GB of edges into Kuzu with 16 GB memory limited failed with out-of-memory error.

To avoid loading a big file into Kuzu in one go, I decided to split it into 1GB-sized partitions. This was done using the following command:
```
split -b 1G /data/project/anonym/ScalingGNNsapers_100M_classification/data/papers100M/processed/edge_index.csv /data/project/anonym/ScalingGNNs/papers_100M_classification/data/papers100M/processed/split_
```

However, this may cause artifacts at the beginning and end of each splitted file as a single CSV line might be split broken between two files. To fix this, I've created a bash script `print_first_last_lines.sh` that prints out the first and last line that can be merged manually.

Finally, the `remove_first_last_lines.sh` needs to be run to delete the first and last line from each csv file. The manually corrected lines can then be inserted in a seperate csv file.

#### **Transform Data into CSV Format**
Some OGBN datasets are automatically downloaded in the NPY format. However, it is easier to loader a CSV into a graph DB. To make the conversion, use the script below, which makes it easier to load into Neo4j and Kuzu.

```bash
sbatch jobs/transform_npy_to_csv.sh
```

### **Running the Models**

#### **Neo4j Backend**

Ensure Neo4j is running, and the ogbn-papers100M dataset is loaded into the database using the script:
```bash
sbatch jobs/load_into_neo4j.sh
```
Then, run the training with the Neo4j backend:
```bash
sbatch jobs/run_neo4j_backend.sh
```

#### **Kuzu Backend**

To use Kuzu, first load the dataset:
```bash
sbatch jobs/load_into_kuzu.sh
```
Then, execute the training script:
```bash
sbatch jobs/run_kuzu_backend.sh
```

#### **In-Memory**

For the traditional in-memory approach, execute:
```bash
sbatch jobs/run_no_backend.sh
```

### **Repository Structure**

- **`kuzu_backend.py`**: Script for Kuzu backend operations.
- **`neo4j_backend.py`**: Script for Neo4j backend operations.
- **`no_backend.py`**: Script for in-memory operations.

- **`src/`**: Core source code for training and backend interaction.
  - `graph_sage.py`: GraphSAGE model architecture for node classification.
  - `graph_sampler.py`: Custom graph sampling for efficient training.
  - `model_v2.py`: Second version of the model implementation.
  - `train_test.py`: Core script for training and evaluation.
  - `utils.py`: Utility functions for configuration reading.

- **`config/`**: Contains configuration files for different backends and models.
  - `loader_params.json`: Parameters for loading the dataset.
  - `model_params.json`: Parameters for the model architecture and training.

- **`jobs/`**: Shell scripts to manage data loading, transformation, and running training jobs.
  - `gpu_run_kuzu.sh`: Script for running Kuzu backend on GPU.
  - `gpu_run_no_backend.sh`: Script for running in-memory backend on GPU.
  - `load_into_kuzu.sh`: Script to load ogbn-papers100M into Kuzu.
  - `load_into_neo4j.sh`: Script to load ogbn-papers100M into Neo4j.
  - `run_kuzu_backend.sh`: Script to run GNN training with Kuzu backend.
  - `run_neo4j_backend.sh`: Script to run GNN training with Neo4j backend.
  - `run_no_backend.sh`: Script to run in-memory training.

- **`loaders/`**: Scripts for loading and manipulating the ogbn-papers100M dataset.
  - `load_into_kuzu.py`: Loads the dataset into Kuzu.
  - `load_into_neo4j.py`: Loads the dataset into Neo4j.
  - `transform_npy_to_csv.py`: Transforms data from `.npy` format to CSV.
  
- **`plots/`**: Contains visualizations of model performance and other metrics.
  - `loss_plot.png`: Loss over training epochs.
  - `time_plot.png`: Time taken per epoch for different backends.

## **References**

For more information on the methods and results, please refer to the report on scaling GNNs with graph databases.
