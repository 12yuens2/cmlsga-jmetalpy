#!/bin/bash

# Set resource requirements

#SBATCH --array=0-8
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=60:00:00

# Logging
#test 

#SBATCH --output=logs/unsgaiii.out
#SBATCH --error=logs/unsgaiii.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

cd $HOME/cmlsga-jmetalpy

# load environment
module load python/3.7.3
source env/bin/activate
#n=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

#run experiment
python src/run_experiment.py corrections/parameters/extra/unsgaiii$SLURM_ARRAY_TASK_ID
