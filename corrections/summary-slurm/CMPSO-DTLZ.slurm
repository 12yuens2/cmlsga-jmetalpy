#!/bin/bash
#SBATCH --array=1-7
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:55:00

# Logging
#SBATCH --output=../logs/CMPSO-DTLZ.out
#SBATCH --error=../logs/CMPSO-DTLZ.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/generate_visualisations.py /ssdfs/users/sy6u19/data-100000evals-30runs-corrections-normals/CMPSO/DTLZ${SLURM_ARRAY_TASK_ID} 
