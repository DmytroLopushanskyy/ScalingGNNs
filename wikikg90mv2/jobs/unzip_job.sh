#!/bin/bash
#SBATCH --job-name=unzip_large_archive     # Job name
#SBATCH --partition=long                 # Partition name
#SBATCH --ntasks=1                         # Number of tasks (1 for single task)
#SBATCH --cpus-per-task=2                  # Number of CPU cores per task (since unzip is single-threaded)
#SBATCH --mem=32G                          # Memory per node
#SBATCH --time=24:00:00                    # Maximum runtime (adjust as needed)
#SBATCH --output=unzip_output.log          # Output log file
#SBATCH --error=unzip_error.log            # Error log file

# Directory paths
ZIP_FILE="/data/coml-intersection-joins/kebl7757/WIKIKG90MV2/wikikg90m-v2.zip"
DEST_DIR="/data/coml-intersection-joins/kebl7757/WIKIKG90MV2/wikikg90m-v2"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

# Extract all files sequentially
echo "Starting extraction of all files from $ZIP_FILE to $DEST_DIR"

unzip -o "$ZIP_FILE" -d "$DEST_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
  echo "Error: Extraction failed." >&2
  exit 1
fi

echo "Extraction completed successfully."
