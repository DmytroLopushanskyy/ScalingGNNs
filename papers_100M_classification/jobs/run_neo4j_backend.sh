#!/bin/bash
#SBATCH --job-name=run_neo4j          # Job name
#SBATCH --partition=long              # Partition to submit to
#SBATCH --ntasks=1                    # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --mem=128GB                   # Memory per node
#SBATCH --time=24:00:00               # Maximum runtime
#SBATCH --output=logs/run_neo4j_output.log  # Output log file
#SBATCH --error=logs/run_neo4j_error.log    # Error log file

# Start neo4j instance
/data/coml-intersection-joins/kebl7757/neo4j-community-5.21.0/bin/neo4j start

# Get Neo4j process ID
NEO4J_PID=$(pgrep -f 'neo4j')

# Wait for 5 minutes, reporting CPU and memory usage every 10 seconds
echo "Starting Neo4j and monitoring system resources for 5 minutes..."
for i in {1..30}
do
  if [ -n "$NEO4J_PID" ]; then
    echo "Resource usage at $(date):"
    top -b -n1 -p $NEO4J_PID | grep $NEO4J_PID | awk '{print "CPU Usage: " $9 "%, Memory Usage: " $10 " GB"}'
  else
    echo "Neo4j process not found."
  fi
  sleep 10s
done

# Capture logs
echo "Neo4j startup logs:"
tail -n 50 /data/coml-intersection-joins/kebl7757/neo4j-community-5.21.0/logs/neo4j.log

# Stop neo4j instance
/data/coml-intersection-joins/kebl7757/neo4j-community-5.21.0/bin/neo4j stop

# Wait for 3 minutes, reporting CPU and memory usage every 10 seconds
echo "Neo4j stopped. Monitoring system resources for another 3 minutes..."
for i in {1..18}
do
  if [ -n "$NEO4J_PID" ]; then
    echo "Resource usage at $(date):"
    top -b -n1 -p $NEO4J_PID | grep $NEO4J_PID | awk '{print "CPU Usage: " $9 "%, Memory Usage: " $10 " GB"}'
  else
    echo "Neo4j process not found."
  fi
  sleep 10s
done

echo "Job completed."

# End of script