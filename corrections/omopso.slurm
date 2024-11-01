#!/bin/bash

# Set resource requirements

#SBATCH --array=0-10
#SBATCH --mem=4000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=34:05:00

# Logging
#test 

#SBATCH --output=logs/omopso.out
#SBATCH --error=logs/omopso.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

cd $HOME/cmlsga-jmetalpy

# load environment
module load python/3.7.3
source env/bin/activate
#n=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

#run experiment
python src/run_experiment.py corrections/parameters/omopso$SLURM_ARRAY_TASK_ID
