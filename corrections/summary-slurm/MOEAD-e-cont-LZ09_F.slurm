#!/bin/bash
#SBATCH --array=1-9
#SBATCH --mem=8000
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=47:55:00

# Logging
#SBATCH --output=../logs/MOEAD-e-cont-LZ09_F.out
#SBATCH --error=../logs/MOEAD-e-cont-LZ09_F.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sy6u19@soton.ac.uk

# load environment
cd $HOME/cmlsga-jmetalpy
module load python/3.7.3
source env/bin/activate

python src/generate_visualisations.py /ssdfs/users/sy6u19/data-100000evals-30runs-corrections-normals/MOEAD-e-cont/LZ09_F${SLURM_ARRAY_TASK_ID} 
