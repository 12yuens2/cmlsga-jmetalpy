#!/bin/bash
#SBATCH --array=1-9
#SBATCH --mem=3000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=15:05:00

# Logging
#SBATCH --output=/scratch/sy6u19/batch-output/smac-cmpso-WFG.out
#SBATCH --error=/scratch/sy6u19/batch-output/smac-cmpso-WFG.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/tuning.py cmpso WFG${SLURM_ARRAY_TASK_ID} pso 100000 5
