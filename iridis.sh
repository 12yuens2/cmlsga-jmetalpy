#!/bin/bash
#PBS -l walltime=18:00:00
#PBS -l nodes=2:ppn=4

# qsub command:
# qsub -v PARAMETERS=parameters.json iridis.sh


# Script to run code on iridis

cd $PBS_O_WORKDIR

echo "Running with params $PARAMETERS"
cat $PARAMETERS

module load conda/4.4.0
source env/bin/activate
python src/run_experiment.py $PARAMETERS
