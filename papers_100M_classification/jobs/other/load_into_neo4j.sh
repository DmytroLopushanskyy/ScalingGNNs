#!/bin/bash
#SBATCH --job-name=load_into_neo4j   # Job name
#SBATCH --partition=long            # Partition to submit to
#SBATCH --ntasks=1                  # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=32          # Number of CPU cores per task
#SBATCH --mem=128GB                  # Memory per node
#SBATCH --time=24:00:00             # Maximum runtime
#SBATCH --output=logs/load_into_neo4j_output.log          # Output log file
#SBATCH --error=logs/load_into_neo4j_error.log            # Error log file

DATA_PATH=/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/data
NEO4J_HOME=/data/coml-intersection-joins/kebl7757/neo4j-fixed-ids-and-features

export $NEO4J_HOME

# Import data into Neo4j
$NEO4J_HOME/bin/neo4j-admin database import full papers100m \
    --nodes=PAPER=$DATA_PATH/papers100M/nodes_split/nodes_header.csv,data/papers100M/nodes_split/nodes_chunk_.* \
    --relationships=CITES=$DATA_PATH/papers100M/edge_idx_split/_edge_header.csv,data/papers100M/edge_idx_split/split.* \
    --normalize-types=false \
    --overwrite-destination=true \
    --max-off-heap-memory=99% \
    --read-buffer-size=16777216


