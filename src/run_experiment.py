import os
import json
import sys

from functools import partial

from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import *
from jmetal.problem.multiobjective.constrained import Srinivas
from jmetal.problem.multiobjective.zdt import *
from jmetal.problem.multiobjective.dtlz import *
from jmetal.problem.multiobjective.lz09 import *
from jmetal.problem.multiobjective.fda import *
from jmetal.lab.experiment import *
from jmetal.util.evaluator import MapEvaluator, SparkEvaluator
from jmetal.util.observer import ProgressBarObserver

from cmlsga.mls import MultiLevelSelection
from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *
from cmlsga.problems.uf import *
from cmlsga.problems.wfg import *
from cmlsga.problems.dascmop import *
from cmlsga.problems.cdf import *
from cmlsga.problems.udf import *
from cmlsga.problems.jy import *
from cmlsga.problems.imb import *
from cmlsga.problems.mop import *

class IncrementalOutputJob(Job):

    def __init__(self, algorithm, algorithm_tag, problem_tag, run):
        super(IncrementalOutputJob, self).__init__(algorithm, algorithm_tag, problem_tag, run)
        self.algorithm.run_tag = run

    def execute(self, output_path):
        def incremental_output(evaluations):
            if output_path:
                file_name = os.path.join(output_path, 'FUN.{}.tsv.{}'.format(self.run_tag, evaluations))
                print_function_values_to_file(self.algorithm.get_result(), filename=file_name)

                file_name = os.path.join(output_path, 'VAR.{}.tsv.{}'.format(self.run_tag, evaluations))
                print_variables_to_file(self.algorithm.get_result(), filename=file_name)

        self.algorithm.output_job = incremental_output
        self.algorithm.output_path = output_path

        super().execute(output_path)


def configure_experiment(population_size, max_evaluations, number_of_runs,
                         algorithms, problems):
    jobs = []

    e = MapEvaluator(processes=4)
    for run in range(number_of_runs):
        for problem in problems:
            for algorithm in algorithms:
                constructor, kwargs = algorithm(problem, population_size,
                                                max_evaluations, e)
                algorithm = constructor(**kwargs)
                algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

                #for dynamic problems
                if isinstance(problem, DynamicProblem):
                    algorithm.observable.register(problem)
                
                jobs.append(IncrementalOutputJob(
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


def print_usage():
    print("Usage:")
    print("python {} `parameters.json`".format(sys.argv[0]))


def parse_algorithms(parameters):
    # convert string names of algorithms into functions with globals()

    algorithms = [globals()[name] for name in parameters["algorithms"]]
    if parameters["mlsga"]:
        collectives = [globals()[algorithm] for algorithm in parameters["mlsga"]]
        algorithms.append(partial(mlsga, collectives))

    return algorithms


def parse_problems(parameters):
    problems = []
    for problem in parameters["problems"]:
        problem_string = problem.split("(")

        if len(problem_string) == 1:
            problems.append(globals()[problem]())
        else: # DASCMOP case
            difficulty = int(problem_string[1][0])
            problems.append(globals()[problem_string[0]](difficulty))

    return problems



if __name__ == "__main__":

    if len(sys.argv) == 2:
        json_file = open(sys.argv[1], "r")
        parameters = json.loads(json_file.read())

        population_size = parameters["population_size"]
        max_evaluations = parameters["max_evaluations"]
        number_of_runs = parameters["number_of_runs"]
        comment = parameters["comment"]

        algorithms = parse_algorithms(parameters)
        problems = parse_problems(parameters)

        run_experiment(population_size, max_evaluations, number_of_runs, comment=comment,
                       algorithms=algorithms,
                       problems=problems)
    else:
        print_usage()
