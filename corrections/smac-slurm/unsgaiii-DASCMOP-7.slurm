#!/bin/bash
#SBATCH --array=1-6
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=49:55:00

# Logging
#SBATCH --output=../logs/smac-unsgaiii-DASCMOP-7.out
#SBATCH --error=../logs/smac-unsgaiii-DASCMOP-7.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/tuning.py unsgaiii DASCMOP${SLURM_ARRAY_TASK_ID}\(7\) 
