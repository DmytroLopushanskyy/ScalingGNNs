# **Scaling Graph Neural Networks With Graph Databases**

This repository contains multiple projects focused on scaling Graph Neural Networks (GNNs) using remote backends like Neo4j, Kuzu, and in-memory processing. Each project addresses different datasets, such as **ogbn-products**, **ogbn-papers100M**, and more, showcasing various approaches to handling large-scale graph data.

### **Structure of the Repository**

Each project in this repository tackles a specific dataset with its own implementation and backend setup. You can find **individual README files** within each folder for detailed instructions on how to run the models, configure environments, and process data for each specific task. These README files cover information like:
- How to load datasets into backends (Neo4j, Kuzu).
- Running GNN models using remote and in-memory backends.
- Custom environment setup for each project.
- Training and evaluation scripts for the specific dataset.

### **Environment Setup**

To set up the environment for running the various projects, use the following steps:

1. Install PyTorch and the required packages:
```bash
conda install pytorch==2.3.0 torchvision torchaudio cpuonly -c pytorch
```
2. Install CMake and GCC:
```bash
conda install -c conda-forge cmake
conda install -c conda-forge gcc=9 gxx=9
```
3. Install `pyg-lib` for efficient sampling:
```bash
pip install git+https://github.com/pyg-team/pyg-lib.git@0.4.0
```

### **Custom Sampler**

For the Neo4j backend, you need to replace the default `neighbor_sampler.py` in the PyTorch Geometric installation. The file is located at:
```bash
/home/anonym/miniconda3/envs/CONDA_ENV_NAME/lib/python3.9/site-packages/torch_geometric/sampler/neighbor_sampler.py
```
Use the custom sampler ```torch_geom_neighbor_sampler.py``` provided in the repository for optimal sampling performance during training.