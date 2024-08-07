#!/bin/bash
#SBATCH --job-name=replace_delimiters   # Job name
#SBATCH --partition=long               # Partition to submit to
#SBATCH --ntasks=1                      # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=4               # Number of CPU cores per task
#SBATCH --mem=16GB                      # Memory per node
#SBATCH --time=12:00:00                 # Maximum runtime
#SBATCH --output=logs/replace_output.log # Output log file
#SBATCH --error=logs/replace_error.log   # Error log file

# Define data path
DATA_PATH=/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/data/papers100M/nodes_split

# Change to temporary directory
cd $SCRATCH || exit 1

# Create a temporary directory for the data
mkdir -p $SCRATCH/nodes_split

# Copy files to $SCRATCH
rsync -av $DATA_PATH/ $SCRATCH/nodes_split/

# Change to the directory containing the data files
cd $SCRATCH/nodes_split || exit 1

# Replace semicolons with commas in each CSV file in nodes_split, except for nodes_chunk_1.csv, nodes_chunk_2.csv, and nodes_chunk_3.csv
for file in nodes_chunk_*.csv; do
  if [[ ! "$file" =~ nodes_chunk_[123]\.csv ]]; then
    echo "Processing $file"
    perl -pi -e 's/;/,/g' "$file"
  fi
done

# Copy modified files back to the original directory
rsync -av $SCRATCH/nodes_split/ $DATA_PATH/
