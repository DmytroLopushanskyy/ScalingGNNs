#!/bin/bash

# Directory containing split CSV files
split_files_path="/data/project/anonym/ScalingGNNs/papers_100M_classification/data/papers100M/edge_idx_split"

# Loop through each split file
for file in "$split_files_path"/split_*.csv; do
    # Check if the file exists to avoid errors
    if [ -e "$file" ]; then
        # Extract the filename without the path
        filename=$(basename "$file")
        
        # Get the first and last lines of the file
        first_line=$(head -n 1 "$file")
        last_line=$(tail -n 1 "$file")
        
        # Print the filename, first line, and last line
        echo "File: $filename"
        echo "First Line: $first_line"
        echo "Last Line: $last_line"
        echo "-----------------------------"
    else
        echo "File: $file does not exist."
    fi
done

