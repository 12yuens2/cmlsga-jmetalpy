#!/bin/bash
# Script to run code on iridis

POPULATION=$1
EVALUATIONS=$2
RUNS=$3
COMMENT=$4

cd $PBS_O_WORKDIR

echo "Running $POPULATION population, $EVALUATIONS evaluations, $RUNS runs, $COMMENT"

python src/run_experiment $POPULATION $EVALUATIONS $RUNS $COMMENT
