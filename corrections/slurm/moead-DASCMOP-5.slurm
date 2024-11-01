#!/bin/bash
#SBATCH --array=1,2
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:55:00

# Logging
#SBATCH --output=../logs/moead-DASCMOP-5.out
#SBATCH --error=../logs/moead-DASCMOP-5.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/run_experiment.py moead DASCMOP${SLURM_ARRAY_TASK_ID}\(5\) 
