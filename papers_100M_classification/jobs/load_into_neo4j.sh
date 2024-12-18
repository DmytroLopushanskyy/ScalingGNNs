#!/bin/bash
#SBATCH --job-name=load_into_neo4j                 # Job name
#SBATCH --partition=long                           # Partition to submit to
#SBATCH --ntasks=1                                 # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=32                         # Number of CPU cores per task
#SBATCH --mem=128GB                                # Memory per node
#SBATCH --time=24:00:00                            # Maximum runtime
#SBATCH --output=logs/load_into_neo4j.log          # Output log file

# Define data path
DATA_PATH=/data/project/anonym/ScalingGNNs/papers_100M_classification/data

# Change to temporary directory
cd $SCRATCH || exit 1

# Copy Neo4j directory to $SCRATCH
rsync -av /data/project/anonym/neo4j $SCRATCH/

# Change to Neo4j directory
cd $SCRATCH/neo4j || exit 1

export NEO4J_HOME=$SCRATCH/neo4j

# Import data into Neo4j
$SCRATCH/neo4j/bin/neo4j-admin database import full papers100m \
    --nodes=PAPER=$DATA_PATH/papers100M/nodes_split/nodes_header.csv,$DATA_PATH/papers100M/nodes_split/nodes_chunk_.* \
    --relationships=CITES=$DATA_PATH/papers100M/edge_idx_split/_edge_header.csv,$DATA_PATH/papers100M/edge_idx_split/split.* \
    --normalize-types=false \
    --overwrite-destination=true \
    --max-off-heap-memory=99% \
    --read-buffer-size=16777216

# Move all files back to original directory
rsync -av --exclude=tmp $SCRATCH/neo4j/ /data/project/anonym/neo4j/