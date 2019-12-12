# cmlsga-jmetalpy

Python implementation of [cMLSGA ](https://github.com/12yuens2/cmlsga) 
built on top of the [jMetalPy](https://github.com/jMetal/jMetalPy) framework.

[![Build Status](https://travis-ci.com/12yuens2/cmlsga-jmetalpy.svg?token=kw2dzDpUGFzFfNgSo4Ns&branch=master)](https://travis-ci.com/12yuens2/cmlsga-jmetalpy)

## Build instructions
Install required packages with
```
pip install -r requirements.txt
```

## Run instructions
Currently there are 2 scripts: `run_experiment.py` and `generate_visualisations.py`

### run_experiment
This script takes 3 inputs: population size, maximum evaluations, and number of runs.

For example, run the script with
```
python src/run_experiment.py 600 30000 30
```
to use a population of 600 individuals, stopping at 30000 function evaluations, for 30 runs.

A csv file with the filename `600pop-30000evals-30runs.csv` will be created in the directory where the script was run.
Detailed data for each algorithm and run will be stored in the `data-600pop-30000evals-30runs` directory.

### generate_visualisations
This script aggregates stats from the csv file created by `run_experiment.py`

Pass the csv file into the script like so:
```
python src/generate_visualisations.py 600pop-30000evals-30runs.csv
```

Multiple csv files can be passed in and statistics/visualisations will be generated for each of them
```
python src/generate_visualisations.py data1.csv data2.csv data3.csv
```

## Acknowledgements
Original idea by Adam Sobey

Written by Przemyslaw Grudniewski

Updates and additional ideas by Przemyslaw Grudniewski

Python tidied up and parallelised by Amy Parkes and Jenny Walker
