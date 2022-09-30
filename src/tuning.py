import numpy as np
import os
import shutil
import time

from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *

from cmlsga.problems.wfg import *
from cmlsga.problems.uf import *
from cmlsga.problems.imb import *

from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.experiment import *
from jmetal.problem.multiobjective.zdt import *
from jmetal.util.evaluator import MapEvaluator

from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, UniformFloatHyperparameter

def tae(cfg):
    #TODO don't hardcode
    max_evaluations = 100000
    runs = 5
    output_path = "data-smac-100k"

    algorithm = globals()[cfg["algorithm"]]
    problem = globals()[cfg["problem"]]()
    population_size = cfg["population"]
    mutation_rate = cfg["mutation"]
    crossover_rate = cfg["crossover"]
    leaders = cfg["leaders"]
    e = MapEvaluator(processes=4)

    constructor, kwargs = algorithm(problem, population_size, max_evaluations, e, mutation_rate, leaders)
    algorithm = constructor(**kwargs)


    jobs = [Job(algorithm, algorithm.get_name(), problem.get_name(), run)
            for run in range(runs)]
    experiment = Experiment(output_path, jobs)
    experiment.run()

    generate_summary_from_experiment(output_path, [InvertedGenerationalDistance(None)],
                                     reference_fronts="resources/reference_front")

    igd = compute_mean_indicator("QualityIndicatorSummary.csv", "IGD")

    # Cleanup
    os.remove("QualityIndicatorSummary.csv")
    shutil.rmtree(output_path)

    # 1 - igd to minimise
    return 1 - igd


if __name__ == "__main__":

    algorithms = ["smpso"]
    problems = [
        #"ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
        #"WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9", 
        #"UF1", "UF2", "UF3", "UF4", "UF5", "UF6", "UF7", "UF8", "UF9"]
        "IMB1", "IMB2", "IMB3", "IMB4", "IMB5", "IMB6", "IMB7", "IMB8", "IMB9",
        "IMB10","IMB11","IMB12","IMB13","IMB14"]

    for algorithm in algorithms:
        for problem in problems:
            cs = ConfigurationSpace()

            # Constants
            cs.add_hyperparameter(Constant("problem", problem))
            cs.add_hyperparameter(Constant("algorithm", algorithm))

            # Parameters to tune
            moead_pops = [100,300,400,500,600,800,1000]
            #cs.add_hyperparameter(CategoricalHyperparameter("population", moead_pops))
            #cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.1, 1.0))
            #cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 0.2))

            # PSO parameters
            cs.add_hyperparameter(CategoricalHyperparameter("population", [i for i in range(100,1100, 100)]))
            cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 0.2))
            cs.add_hyperparameter(CategoricalHyperparameter("leaders", [i for i in range(100, 1100, 100)]))

            scenario = Scenario(
                {
                    "run_obj": "quality",
                    "runcount-limit": 100,
                    "cs": cs,
                }
            )

            smac = SMAC4BB(
                scenario=scenario,
                rng=np.random.RandomState(5),
                tae_runner=tae
            )

            start = time.time()
            best = smac.optimize()
            end = time.time()

            with open("smac-parameters.csv", "a") as smac_output:
                # Algorithm, Problem, Population size, Crossover, Mutation, Time (seconds)
                smac_output.write("{},{},{},{},{},{}, {}\n".format(
                    algorithm, problem, best["population"], best["crossover"], best["mutation"], best["leaders"], end - start
                ))

            print("Algorithm: {}, Problem: {}".format(algorithm, problem))
            print("best: population: {}, crossover: {}, mutation: {}, leaders: {}".format(
                best["population"], best["crossover"], best["mutation"], best["leaders"]
            ))
