#!/bin/bash
#SBATCH --array=1-7
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=49:55:00

# Logging
#SBATCH --output=../logs/smac-smpso-MOP.out
#SBATCH --error=../logs/smac-smpso-MOP.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/tuning.py smpso MOP${SLURM_ARRAY_TASK_ID} 
