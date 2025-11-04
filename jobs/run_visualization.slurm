#!/bin/bash
#SBATCH --job-name=h_visualization
#SBATCH --partition=allgroups
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=outputs/logs/h_visualization_s0_l025_%j.out
#SBATCH --error=outputs/logs/h_visualization_s0_l025_%j.err

echo "Job $SLURM_JOB_ID started at: $(date) on $(hostname)"
echo "Working directory: $(pwd)"

# move to submission directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment (adjust path if needed)
source venv/bin/activate

# Run the code
srun python3 source_code/visualization_hoffman.py 

echo "Job finished at: $(date)"
