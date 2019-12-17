import os
import sys

from functools import partial

from jmetal.core.quality_indicator import *
from jmetal.problem.multiobjective.constrained import Srinivas
from jmetal.problem.multiobjective.zdt import *
from jmetal.lab.experiment import *
from jmetal.util.observer import ProgressBarObserver

from cmlsga.mls import MultiLevelSelection
from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *

def configure_experiment(population_size, max_evaluations, number_of_runs,
                         algorithms, problems):
    jobs = []

    for run in range(number_of_runs):
        for problem in problems:
            for algorithm in algorithms:
                constructor, kwargs = algorithm(problem, population_size,
                                                max_evaluations, store.default_evaluator)
                algorithm = constructor(**kwargs)
                algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

                jobs.append(Job(
                    algorithm=algorithm,
                    algorithm_tag=algorithm.get_name(),
                    problem_tag=problem.get_name(),
                    run=run
                ))

    return jobs


def run_experiment(population_size, max_evaluations, number_of_runs, comment="",
                   algorithms=[partial(mlsga, [nsgaii]),
                               partial(mlsga, [omopso]),
                               nsgaii, moead, omopso],
                   problems=[ZDT1(), ZDT2(), ZDT3(), ZDT4(), ZDT6()]):

    meta = "{}pop-{}evals-{}runs-{}".format(population_size, max_evaluations,
                                            number_of_runs, comment)

    jobs = configure_experiment(population_size, max_evaluations, number_of_runs,
                                algorithms, problems)

    output_directory = "data-{}".format(meta)

    experiment = Experiment(output_directory, jobs)
    experiment.run()

    generate_summary_from_experiment(
        output_directory,
        [InvertedGenerationalDistance(), HyperVolume([1.0, 1.0])],
        reference_fronts="resources/reference_front",
    )

    #rename file
    os.rename("QualityIndicatorSummary.csv", "{}.csv".format(meta))


def print_usage():
    print("Usage:")
    print("python {} [population size] [max evaluations] [number of runs]".format(sys.argv[0]))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print_usage()
    else:
        population_size = int(sys.argv[1])
        max_evaluations = int(sys.argv[2])
        number_of_runs = int(sys.argv[3])
        if sys.argv[4]:
            comment = sys.argv[4]

        run_experiment(population_size, max_evaluations, number_of_runs, comment=comment)
