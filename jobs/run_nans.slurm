#!/bin/bash

#SBATCH --job-name explore_data
#SBATCH --output outputs/logs/nans_%j.txt
#SBATCH --error outputs/logs/nans_%j.txt
#SBATCH --time 30:00
#SBATCH --ntasks 1
#SBATCH --partition allgroups
#SBATCH --mem 8G

# Change directory
cd $SLURM_SUBMIT_DIR

# info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Activate virtual environment
source venv/bin/activate

# Run the script
srun python3 source_code/count_nans.py

echo "Job finished at: $(date)"
