#!/bin/bash
#SBATCH --array=1-6
#SBATCH --mem=4000
#SBATCH --ntasks=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=24:05:00

# Logging
#SBATCH --output=smac-moead-DASCMOP-7.out
#SBATCH --error=smac-moead-DASCMOP-7.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
module load python/3.7.3
source env/bin/activate

cd $HOME/cmlsga-jmetalpy
python src/tuning.py moead DASCMOP${SLURM_ARRAY_TASK_ID}\(7\) ga 100000 5
