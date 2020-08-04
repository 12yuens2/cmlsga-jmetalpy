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
This script takes 1 input: a `parameters.json` file

Run the script with
```
python src/run_experiment.py 600 30000 30
```

Example of parameter file:

``` json
{
    "population_size": 300,
    "max_evaluations": 10000,
    "number_of_runs": 3,
    "comment": "",

    "mlsga": ["nsgaii"],
    "algorithms": ["nsgaii", "moead", "omopso"],
    "problems": ["ZDT1", "ZDT2"]
}
```
- `mlsga` determines if the experiment uses the MLSGA algorithm. Use an empty array to disable using MLSGA. Supported algorithms in this field are:
  - `nsagii`, `moead`, `omopso`, `smpso`, `genetic_algorithm`
- `algorithms`: List of algorithms to use in this experiment, supported algorithms are:
  - `nsgaii`, `moead`, `omopso`, `smpso`
- `problems`: List of problems to use in the experiment, supported problems are:
  - `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`
  - `DTLZ1`, `DTLZ2`, `DTLZ3`, `DTLZ4`, `DTLZ5`, `DTLZ6`, `DTLZ7`
  - `LZ09_F1`, `LZ09_F2`, `LZ09_F3`, `LZ09_F4`, `LZ09_F5`, `LZ09_F6`, `LZ09_F7`, `LZ09_F8`, `LZ09_F9`


A csv file with the filename `600pop-30000evals-30runs.csv` will be created in the directory where the script was run.
Detailed data for each algorithm and run will be stored in the `data-600pop-30000evals-30runs` directory.

### generate_visualisations
This script aggregates stats from the csv file created by `run_experiupment.py`

Pass the data folder created into the script like so:
```
python src/generate_visualisations.py data-600pop-30000evals-30runs-
```


## Acknowledgements
Original idea by Adam Sobey

Written by Przemyslaw Grudniewski

Updates and additional ideas by Przemyslaw Grudniewski

Python tidied up and parallelised by Amy Parkes and Jenny Walker
