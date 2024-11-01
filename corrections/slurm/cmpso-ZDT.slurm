#!/bin/bash
#SBATCH --array=1-4,6
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:55:00

# Logging
#SBATCH --output=../logs/cmpso-ZDT.out
#SBATCH --error=../logs/cmpso-ZDT.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/run_experiment.py cmpso ZDT${SLURM_ARRAY_TASK_ID} 
