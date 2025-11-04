#!/bin/bash
#SBATCH --job-name=hoffman_classify
#SBATCH --partition=allgroups
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --mem=6G
#SBATCH --cpus-per-task=2
#SBATCH --output=outputs/logs/hoffman_s0_l025_%j.out
#SBATCH --error=outputs/logs/hoffman_s0_l025_%j.err

echo "Job $SLURM_JOB_ID started at: $(date) on $(hostname)"
echo "Working directory: $(pwd)"

# move to submission directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment (adjust path if needed)
source venv/bin/activate


# Example arguments (edit as needed)
# SIGMA: smoothing in voxels (or use --sigma_phys to give Mpc/h)
SIGMA=0
# LAMBDA threshold
LAMBDA=0.25
# box size used in preprocessing (Mpc/h)
BOXSIZE=1000.0
# input and output paths
INPATH=data/preprocessed
OUTPATH=data/preprocessed

# Run the classifier
srun python3 source_code/hoffman_classification.py --boxsize ${BOXSIZE} --sigma ${SIGMA} --lambda_th ${LAMBDA} --inpath ${INPATH} --outpath ${OUTPATH}

echo "Job finished at: $(date)"
