#!/bin/bash

# Set resource requirements

#SBATCH --mem=4000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=44:05:00

# Logging
#test 

#SBATCH --output=logs/summary.out
#SBATCH --error=logs/summary.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

cd $HOME/cmlsga-jmetalpy

# load environment
module load python/3.7.3
source env/bin/activate
#n=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

#run experiment
python src/generate_visualisations.py /ssdfs/users/sy6u19/data-100000evals-30runs-corrections-normals/
