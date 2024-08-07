#!/bin/bash
#SBATCH --job-name=run_neo4j_sh          # Job name
#SBATCH --partition=short              # Partition to submit to
#SBATCH --ntasks=1                    # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=48            # Number of CPU cores per task
#SBATCH --mem=256GB                   # Memory per node
#SBATCH --time=12:00:00               # Maximum runtime
#SBATCH --output=logs/neo4j_output.log  # Output log file
#SBATCH --error=logs/neo4j_error.log    # Error log file

# Started with 256 GB, 48 CPUs, medium, 24h - takes forever (more than 8 hours)
# Retrying the same on short, 12h -- all good, logs stored, job finished. 0 neighbours

# Start neo4j instance
/data/coml-intersection-joins/kebl7757/neo4j-community-5.21.0/bin/neo4j start

# Get Neo4j process ID
NEO4J_PID=$(pgrep -f 'neo4j')

# Wait for 5 minutes, reporting CPU and memory usage every 10 seconds
echo "Starting Neo4j and monitoring system resources for 5 minutes..."
for i in {1..5}
do
  sleep 10s
done

# Run file
source /home/kebl7757/miniconda3/etc/profile.d/conda.sh
conda activate single-neo4j

python neo4j_backend.py

# Stop neo4j instance
/data/coml-intersection-joins/kebl7757/neo4j-community-5.21.0/bin/neo4j stop

# Wait for 3 minutes, reporting CPU and memory usage every 10 seconds
echo "Neo4j stopped. Monitoring system resources for another 3 minutes..."
for i in {1..5}
do
  sleep 10s
done

echo "Job completed."

# End of script