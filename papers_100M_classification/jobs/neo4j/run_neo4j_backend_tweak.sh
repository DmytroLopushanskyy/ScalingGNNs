#!/bin/bash
#SBATCH --job-name=run_neo4j_sh_2          # Job name
#SBATCH --partition=short              # Partition to submit to
#SBATCH --ntasks=1                    # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --mem=20GB                   # Memory per node
#SBATCH --time=04:00:00               # Maximum runtime
#SBATCH --output=final_experiments/8_20gb.log  # Output log file

# Define data path
DATA_PATH=/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/data

# Change to temporary directory
cd $SCRATCH || exit 1

# Copy Neo4j directory to $SCRATCH
echo "Starting to copy Neo4j..."
rsync -av /data/coml-intersection-joins/kebl7757/neo4j-community-5.21.0 $SCRATCH/
echo "Copied now!"
# for i in {1..15}
# do
#   sleep 60s
#   echo "waited 1 min."
# done
# echo "Starting!"

# Change to Neo4j directory
cd $SCRATCH/neo4j-community-5.21.0 || exit 1

export NEO4J_HOME=$SCRATCH/neo4j-community-5.21.0

# Import data into Neo4j
$SCRATCH/neo4j-community-5.21.0/bin/neo4j start

# Wait, reporting CPU and memory usage every 10 seconds
echo "Starting Neo4j and monitoring system resources for 50 secs..."
for i in {1..5}
do
  sleep 10s
done

# Run file
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate single-neo4j

cd /data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification
python neo4j_backend.py

# Stop neo4j instance
$SCRATCH/neo4j-community-5.21.0/bin/neo4j stop

echo "Neo4j stopped. Monitoring system resources for another 50 secs..."
for i in {1..5}
do
  sleep 10s
done

echo "Job completed."

# End of script