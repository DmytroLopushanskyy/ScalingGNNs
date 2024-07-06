#!/bin/bash
#SBATCH --job-name=test_fs   # Job name
#SBATCH --partition=short            # Partition to submit to
#SBATCH --ntasks=1                  # Number of tasks (1 for single core job)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --mem=1GB                  # Memory per node
#SBATCH --time=1:00:00             # Maximum runtime
#SBATCH --output=logs/fs_output_2.log          # Output log file
#SBATCH --error=logs/fs_error_2.log            # Error log file

cd $TMPDIR || exit 1

filesystem_type=$(df -T $TMPDIR | awk 'NR==2 {print $2}')

echo "File system type of \$TMPDIR ($TMPDIR): $filesystem_type"