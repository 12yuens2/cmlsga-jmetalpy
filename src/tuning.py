import numpy as np

from cmlsga.algorithms.genetic_algorithms import *

from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.experiment import *
from jmetal.problem.multiobjective.zdt import *
from jmetal.util.evaluator import MapEvaluator

from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

def tae(cfg):
    #TODO don't hardcode
    algorithm = nsgaii
    problem = ZDT1()
    max_evaluations = 10000
    runs = 3
    output_path = "data-smac"

    population_size = cfg["population"]
    mutation_rate = cfg["mutation"]
    crossover_rate = cfg["crossover"]
    e = MapEvaluator(processes=4)

    constructor, kwargs = nsgaii(problem, population_size, max_evaluations, e, mutation_rate, crossover_rate)
    algorithm = constructor(**kwargs)


    jobs = [Job(algorithm, algorithm.get_name(), problem.get_name(), run)
            for run in range(runs)]
    experiment = Experiment(output_path, jobs)
    experiment.run()

    generate_summary_from_experiment(output_path, [InvertedGenerationalDistance(None)],
                                     reference_fronts="resources/reference_front")

    igd = compute_mean_indicator("QualityIndicatorSummary.csv", "IGD")

    # 1 - igd to minimise
    return 1 - igd


if __name__ == "__main__":
    cs = ConfigurationSpace()

    cs.add_hyperparameter(CategoricalHyperparameter("population", choices=[100,200,300,400,500]))
    cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.1, 1.0))
    cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 0.1))

    scenario = Scenario(
        {
            "run_obj": "quality",
            "runcount-limit": 50,
            "cs": cs,
        }
    )


    smac = SMAC4BB(
        scenario=scenario,
        rng=np.random.RandomState(5),
        tae_runner=tae
    )

    best = smac.optimize()

    print(best)
