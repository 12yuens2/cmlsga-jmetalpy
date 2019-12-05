from jmetal.config import store
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.lab.visualization import Plot
from jmetal.problem.multiobjective.constrained import Srinivas
from jmetal.problem.multiobjective.zdt import ZDT1, ZDT2, ZDT6
from jmetal.util.solutions import read_solutions

from mls import MultiLevelSelection
from algorithms.genetic_algorithms import *


def configure_experiment(algorithms, problems, num_runs):
    jobs = []
    max_evaluations = 15000
    population_size = 100

    for run in range(num_runs):
        for problem in problems:
            for algorithm in algorithms:
                nsgaii_constructor, kwargs = nsgaii(problem, population_size,
                                        max_evaluations, store.default_evaluator)
                jobs.append(Job(
                    algorithm=nsgaii_constructor(**kwargs),
                    algorithm_tag="NSGAII",
                    problem_tag=problem.get_name(),
                    run=run
                ))

                moead_constructor, kwargs = moead(problem, population_size,
                                    max_evaluations, store.default_evaluator)
                jobs.append(Job(
                    algorithm=moead_constructor(**kwargs),
                    algorithm_tag="MOEAD",
                    problem_tag=problem.get_name(),
                    run=run
                ))

    return jobs


def experiment():
    jobs = configure_experiment([Srinivas(), ZDT1(), ZDT2()], 30)

    output_directory = "data"
    algorithms = [nsgaii, moead]

    experiment = Experiment(algorithms, output_directory, jobs)
    experiment.run()

    generate_summary_from_experiment(
        output_directory,
        [InvertedGenerationalDistance(), HyperVolume([1.0, 1.0])],
        reference_fronts="resources/reference_front",
    )



def run_algorithm(algorithm, problem,
                  population_size=600, max_evaluations=30000, evaluator=store.default_evaluator):

    constructor, kwargs = algorithm(problem, population_size, max_evaluations, evaluator)
    algo = constructor(**kwargs)
    algo.run()

    return algo.get_result()


def run_mlsga():
    problem = ZDT6()
    problem.reference_front = read_solutions(filename="resources/reference_front/ZDT6.pf")

    population_size = 600
    max_evaluations = 100000

    algorithms = [mlsga, nsgaii, moead, omopso]
    results = {}
    for algorithm in algorithms:
        print("Running {}".format(algorithm.__name__))
        front = run_algorithm(algorithm, problem, population_size, max_evaluations)
        results[algorithm.__name__] = front


    print(results)
    plot_front = Plot(plot_title='Pareto front approximation',
                      reference_front=problem.reference_front,
                      axis_labels=problem.obj_labels)
    plot_front.plot(
        list(results.values()), list(results.keys()), filename=problem.get_name())


if __name__ == "__main__":
    #experiment()
    while True:
        try:
            run_mlsga()
        except Exception as e:
            print(e)
            print("\n")
            continue
        break
