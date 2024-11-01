import numpy as np
import os
import shutil
import time
import sys
import json

from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *

from cmlsga.problems.wfg import *
from cmlsga.problems.uf import *
from cmlsga.problems.imb import *
from cmlsga.problems.dascmop import *
from cmlsga.problems.mop import *
from cmlsga.problems.udf import *
from cmlsga.problems.jy import *
from cmlsga.problems.cdf import *

from jmetal.core.quality_indicator import InvertedGenerationalDistance, HyperVolume
from jmetal.lab.experiment import *
from jmetal.problem.multiobjective.zdt import *
from jmetal.problem.multiobjective.dtlz import *
from jmetal.problem.multiobjective.lz09 import *
from jmetal.problem.multiobjective.fda import *

from jmetal.util.evaluator import MapEvaluator

from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import Constant, CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter


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



def generate_summary_from_experiment_dynamic(input_dir, quality_indicators, problems, evaluations,
                                     reference_fronts = '', summary_filename="QualityIndicatorSummary.csv"):
    reference_change = 5000
    ref_time = 1
    if not quality_indicators:
        quality_indicators = []
        
    with open(summary_filename, 'w+') as of:
        of.write('Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue\n')

    for dirname, _, filenames in os.walk(input_dir):
        print(dirname)
        for filename in sorted(filenames):
            try:
                # Linux filesystem
                algorithm, problem = dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split('\\')[-2:]

            for problem_name in problems:
                results = {}
                if 'FUN' in filename and problem == problem_name:
                    solutions = read_solutions(os.path.join(dirname, filename))
                    digits = [s for s in filename.split('.') if s.isdigit()]
                    run_tag = digits[0]
                    evaluation_tag = evaluations

                    if len(digits) > 1:
                        evaluation_tag = digits[1]

                    if not run_tag in results:
                        results[run_tag] = 0

                    for indicator in quality_indicators:
                        ref_time = min(int(int(evaluation_tag)/reference_change) + 1, 20)
                        reference_front_file = "resources/reference_front/{}_time{}.pf".format(problem_name, ref_time) 
  
                        # Add reference front if any
                        if hasattr(indicator, 'reference_front'):
                            if Path(reference_front_file).is_file():
                                reference_front = []
                                with open(reference_front_file) as file:
                                    for line in file:
                                        reference_front.append([float(x) for x in line.split()])

                                indicator.reference_front = reference_front
                            elif Path("resources/reference_front/{}.pf".format(problem_name)).is_file():
                                reference_front = []
                                with open("resources/reference_front/{}.pf".format(problem_name)) as file:
                                    for line in file:
                                        reference_front.append([float(x) for x in line.split()])

                                indicator.reference_front = reference_front
                            else:
                                print("no reference front for {}".format(problem))

                        results[run_tag] += indicator.compute([solutions[i].objectives for i in range(len(solutions))])

                # Save quality indicator value to file
                for run_tag,igds in results.items():
                    with open(summary_filename, 'a+') as of:
                        of.write(','.join([algorithm, problem, run_tag, indicator.get_short_name(), str(igds / 20.0)]))
                        of.write('\n')


def generate_summary_from_experiment(input_dir, quality_indicators, reference_fronts="", summary_filename="QualityIndicatorSummary.csv"):
    if not quality_indicators:
        quality_indicators = []

    with open(summary_filename, "w+") as of:
        of.write("Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue\n")

    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            try:
                # Linux filesystem
                algorithm, problem = dirname.split("/")[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split("\\")[-2:]

            if "TIME" in filename:
                run_tag = [s for s in filename.split(".") if s.isdigit()].pop()

                with open(os.path.join(dirname, filename), "r") as content_file:
                    content = content_file.read()

                with open(summary_filename, "a+") as of:
                    of.write(",".join([algorithm, problem, run_tag, "Time", str(content)]))
                    of.write("\n")

            if "FUN" in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                run_tag = [s for s in filename.split(".") if s.isdigit()].pop()
                for indicator in quality_indicators:
                    reference_front_file = os.path.join(reference_fronts, problem + ".pf")

                    # Add reference front if any
                    if hasattr(indicator, "reference_front"):
                        if Path(reference_front_file).is_file():
                            reference_front = []
                            with open(reference_front_file) as file:
                                for line in file:
                                    reference_front.append([float(x) for x in line.split()])

                            indicator.reference_front = reference_front
                        else:
                            logger.warning("Reference front not found at", reference_front_file)

                    result = indicator.compute([solutions[i].objectives for i in range(len(solutions))])

                    # Save quality indicator value to file
                    with open(summary_filename, "a+") as of:
                        of.write(",".join([algorithm, problem, run_tag, indicator.get_short_name(), str(result)]))
                        of.write("\n")

def tae(cfg):
    max_evaluations = cfg["evaluations"]
    output_path = cfg["outpath"]

    algorithm_name = cfg["algorithm"]
    algorithm = globals()[cfg["algorithm"]]
    problem_name = cfg["problem"]
    problem = parse_problem(cfg["problem"])

    e = MapEvaluator(processes=2)
    constructor = 0
    kwargs = 0

    population = cfg["population"]

    if algorithm_name == "unsgaiii":
        crossover = cfg["crossover"]
        mutation = cfg["mutation"]
        ref_points = cfg["ref_points"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, mutation, crossover, ref_points)

    if algorithm_name == "moead":
        crossover = cfg["crossover"]
        mutation = cfg["mutation"]
        theta = cfg["neighbour_selection"]
        T = cfg["neighbour_size"]
        F = cfg["F"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, mutation, crossover, F, T, theta)

    if algorithm_name == "heia":
        crossover = cfg["crossover"]
        mutation = cfg["mutation"]
        theta = cfg["theta"]
        NA = cfg["NA"]
        #F = cfg["F"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, mutation, crossover, NA, theta)

    if algorithm_name == "cmlsga":
        crossover = cfg["crossover"]
        mutation = cfg["mutation"]
        collectives = cfg["collectives"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, mutation, crossover, collectives)

    if algorithm_name == "ibea":
        crossover = cfg["crossover"]
        mutation = cfg["mutation"]
        kappa = cfg["kappa"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, mutation, crossover, kappa)

    if algorithm_name == "omopso":
        uniform_mutation = cfg["uniform_mutation"]
        nonuniform_mutation = cfg["nonuniform_mutation"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, uniform_mutation, nonuniform_mutation)

    if algorithm_name == "smpso":
        mutation = cfg["mutation"]

        constructor, kwargs = algorithm(problem, population, max_evaluations, e, mutation)

    if algorithm_name == "cmpso":
        constructor, kwargs = algorithm(problem, population, max_evaluations, e)


    #epigenetic constructor
    #algorithm = constructor(epigenetic_proba, block_size, **kwargs)

    #non-epigenetic constructor
    algorithm = constructor(**kwargs)

    job = Job(algorithm, algorithm.get_name(), problem.get_name(), 1)

    experiment = Experiment(output_path, [job])
    experiment.run()

    summary_filename = output_path + "/QualityIndicatorSummary.csv"

    #dynamic output
    #generate_summary_from_experiment_dynamic(output_path, [InvertedGenerationalDistance(None), HyperVolume(None)], [problem.get_name()], max_evaluations,
    #                                 reference_fronts="resources/reference_front",
    #                                 summary_filename=summary_filename)

    #non-dynamic output
    generate_summary_from_experiment(output_path, [InvertedGenerationalDistance(None)],
                                     reference_fronts="resources/reference_front",
                                     summary_filename=summary_filename)

    igd = compute_mean_indicator(summary_filename, "IGD")

    # Cleanup
    os.remove(summary_filename)
    shutil.rmtree(output_path)

    return igd # We want to minimize


def create_configuration(algorithm_name, problem_name, outpath, evaluations):
    cs = ConfigurationSpace()

    # Constants
    cs.add_hyperparameter(Constant("problem", problem_name))
    cs.add_hyperparameter(Constant("algorithm", algorithm_name))
    cs.add_hyperparameter(Constant("outpath", outpath))
    cs.add_hyperparameter(Constant("evaluations", evaluations))

    # Parameters to tune
    pops = [i for i in range(10, 1002, 2)]

    #cs.add_hyperparameter(UniformFloatHyperparameter("epigenetic_proba", 0.0, 1.0))
    #cs.add_hyperparameter(UniformIntegerHyperparameter("block_size", 1, parse_problem(problem).number_of_variables))

    problem = parse_problem(problem_name)

    # nsgaiii
    if algorithm_name == "unsgaiii":
        cs.add_hyperparameter(UniformIntegerHyperparameter("population", 2, 1000, default_value=128))
        cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.0, 1.0, default_value=0.9))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))
        cs.add_hyperparameter(UniformIntegerHyperparameter("ref_points", 2, 100, default_value=16))

    # moead-de
    if algorithm_name == "moead":
        moead_pops = [100,300,400,500,600,800,1000]
        cs.add_hyperparameter(CategoricalHyperparameter("population", moead_pops, default_value=128))
        cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.0, 1.0, default_value=0.9))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))

        cs.add_hyperparameter(UniformFloatHyperparameter("neighbour_selection", 0.1, 1.0, default_value=0.9))
        cs.add_hyperparameter(UniformIntegerHyperparameter("neighbour_size", 2, 100, default_value=20))
        cs.add_hyperparameter(UniformFloatHyperparameter("F", 0.1, 1.0, default_value=0.5))

    # heia
    if algorithm_name == "heia":
        cs.add_hyperparameter(CategoricalHyperparameter("population", pops, default_value=100))
        cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.0, 1.0, default_value=1.0))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))

        cs.add_hyperparameter(UniformIntegerHyperparameter("NA", 10, 100, default_value=20))
        cs.add_hyperparameter(UniformFloatHyperparameter("theta", 0.1, 1.0, default_value=0.9))
        #cs.add_hyperparameter(UniformFloatHyperparameter("F", 0.1, 1.0, default_value=0.5))

    # cmlsga
    if algorithm_name == "cmlsga":
        cs.add_hyperparameter(CategoricalHyperparameter("population", pops, default_value=100))
        cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.0, 1.0, default_value=1.0))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))

        cs.add_hyperparameter(UniformIntegerHyperparameter("collectives", 2, 64, default_value=8))

    # ibea
    if algorithm_name == "ibea":
        cs.add_hyperparameter(UniformIntegerHyperparameter("population", 2, 1000, default_value=100))
        cs.add_hyperparameter(UniformFloatHyperparameter("crossover", 0.0, 1.0, default_value=1.0))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))
        cs.add_hyperparameter(UniformFloatHyperparameter("kappa", 0.01, 1.0, default_value=0.05))

    # omopso
    if algorithm_name == "omopso":
        cs.add_hyperparameter(UniformIntegerHyperparameter("population", 10, 1000, default_value=100))
        cs.add_hyperparameter(UniformFloatHyperparameter("uniform_mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))
        cs.add_hyperparameter(UniformFloatHyperparameter("nonuniform_mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))

    # smpso
    if algorithm_name == "smpso":
        cs.add_hyperparameter(UniformIntegerHyperparameter("population", 10, 1000, default_value=100))
        cs.add_hyperparameter(UniformFloatHyperparameter("mutation", 0.0, 1.0, default_value=1.0/problem.number_of_variables))

    # cmpso
    if algorithm_name == "cmpso":
         cs.add_hyperparameter(UniformIntegerHyperparameter("population", 2, 1000, default_value=20))


    return cs


def parse_problem(problem):
    problem_string = problem.split("(")

    if len(problem_string) == 1:
        return globals()[problem]()

    else: #DASCMOP case
        difficulty = int(problem_string[1][0])
        return globals()[problem_string[0]](difficulty)

    

# Usage:
# python tuning.py [algorithm] [problem]
if __name__ == "__main__":

    print("start")

    algorithm = sys.argv[1]
    problem = sys.argv[2]
    evaluations = 100000 #int(sys.argv[4])

    mainpath = "/ssdfs/users/sy6u19/corrections-smac/{}/{}".format(algorithm, problem)
    outpath = mainpath + "/data"

    cs = create_configuration(algorithm, problem, outpath, evaluations)

    scenario = Scenario(
	{
	    "output_dir": mainpath,
	    "run_obj": "quality",
	    "cs": cs,
	    "shared_model": True,
	    "input_psmac_dirs": mainpath,
            "wallclock_limit": 86400
	}
    )

    smac = SMAC4AC(
	scenario=scenario,
	rng=np.random.RandomState(5),
	tae_runner=tae,
    )

    best = smac.optimize()

    with open("corrections/smac/{}.csv".format(algorithm), "a") as smac_output:
	# Algorithm, Problem, Population size, Crossover, Mutation, Leaders (for pso), neighour size, selection, F, Epigenetic Prob, Block size
        smac_output.write("{}\n{}\n".format(best.get_dictionary().keys(), best.get_dictionary().values()))

    #print("Algorithm: {}, Problem: {}".format(algorithm, problem))
    #print("best: population: {}, crossover: {}, mutation: {}, leaders: {}".format(
    #    best["epigenetic_proba"], best["block_size"]
    #))

