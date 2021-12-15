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
python src/run_experiment.py parameters.json
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
- `mlsga` determines if the experiment uses the MLSGA algorithm. Use an empty array to disable using MLSGA. Use multiple algorithms here for cMLSGA. Supported algorithms in this field are:
  - `nsagii`, `nsgaiii`, `moead`, `heia`, `omopso`, `smpso`, `cmpso`, `genetic_algorithm`
- `algorithms`: List of algorithms to use in this experiment, supported algorithms are:
  - `nsgaii`, `nsgaiii`, `moead`, `ibea`, `spea2`, `omopso`, `smpso`, `cmpso`, `heia`
- `problems`: List of problems to use in the experiment, supported problems are:
  - `ZDT1`, `ZDT2`, `ZDT3`, `ZDT4`, `ZDT6`
  - `DTLZ1`, `DTLZ2`, `DTLZ3`, `DTLZ4`, `DTLZ5`, `DTLZ6`, `DTLZ7`
  - `LZ09_F1`, `LZ09_F2`, `LZ09_F3`, `LZ09_F4`, `LZ09_F5`, `LZ09_F6`, `LZ09_F7`, `LZ09_F8`, `LZ09_F9`
  - `UF1`, `UF2`, `UF3`, `UF4`, `UF5`, `UF6`, `UF7`, `UF8`, `UF9`, `UF10`
  - `WFG1`, `WFG2`, `WFG3`, `WFG4`, `WFG5`, `WFG6`, `WFG7`, `WFG8`, `WFG9`
  - `IMB1`, `IMB2`, `IMB3`, `IMB4`, `IMB5`, `IMB6`, `IMB7`, `IMB8`, `IMB9`, `IMB10`, `IMB11`, `IMB12`, `IMB13`, `IMB14`
  - `MOP1`, `MOP2`, `MOP3`, `MOP4`, `MOP5`, `MOP6`, `MOP7`
  - `DASCMOP`
  - `FDA1`, `FDA2`, `FDA3`, `FDA4`, `FDA5`
  - `UDF1`, `UDF2`, `UDF3`, `UDF4`, `UDF5`, `UDF6`, `UDF8`, `UDF9`
  - `CDF1`, `CDF2`, `CDF3`, `CDF4`, `CDF5`, `CDF6`, `CDF7`, `CDF8`, `CDF9`, `CDF10`, `CDF11`, `CDF12`, `CDF13`, `CDF14`, `CDF15`
  - `JY1`, `JY2`, `JY3`, `JY4`, `JY5`, `JY6`, `JY7`, `JY8`

Detailed data for each algorithm and run will be stored in the `data-300pop-10000evals-3runs-` directory.

### generate_visualisations
This script aggregates stats from the csv file created by `run_experiment.py`

Pass the data folder created into the script like so:
```
python src/generate_visualisations.py data-600pop-30000evals-30runs-
```

A csv file with the filename `300pop-10000evals-3runs.csv` will be created in the directory where the script was run.



## Acknowledgements
Original idea by Adam Sobey

Written by Przemyslaw Grudniewski

Updates and additional ideas by Przemyslaw Grudniewski

Python tidied up and parallelised by Amy Parkes and Jenny Walker
