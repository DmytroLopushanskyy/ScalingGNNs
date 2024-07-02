#!/bin/bash

# Directory containing split CSV files
split_files_path="/data/coml-intersection-joins/kebl7757/ScalingGNNs/papers_100M_classification/data/papers100M/edge_idx_split"

# Loop through each split file
for file in "$split_files_path"/split_*.csv; do
    # Remove the first and last lines directly
    sed -i '1d;$d' "$file"
    echo "Processed $file"
done

echo "All files processed."

