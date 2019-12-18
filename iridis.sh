#!/bin/bash
#PBS -l walltime=10:00:00
#PBS -l nodes=6:ppn=10

# Script to run code on iridis

#POPULATION=$1
#EVALUATIONS=$2
#RUNS=$3
#COMMENT=$4

cd $PBS_O_WORKDIR

echo "Running $POPULATION population, $EVALUATIONS evaluations, $RUNS runs, $COMMENT"

module load conda/4.4.0
source env/bin/activate
python src/run_experiment.py $POPULATION $EVALUATIONS $RUNS $COMMENT
