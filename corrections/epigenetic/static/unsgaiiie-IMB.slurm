#!/bin/bash
#SBATCH --array=1-14
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:55:00

# Logging
#SBATCH --output=../logs/unsgaiiie-IMB.out
#SBATCH --error=../logs/unsgaiiie-IMB.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/run_experiment.py unsgaiiie IMB${SLURM_ARRAY_TASK_ID} 
