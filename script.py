from jmetal.config import store
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import *
from jmetal.lab.visualization import Plot
from jmetal.problem.multiobjective.constrained import Srinivas
from jmetal.problem.multiobjective.zdt import ZDT1, ZDT2, ZDT6
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions import read_solutions

from mls import MultiLevelSelection
from algorithms.genetic_algorithms import *


def configure_experiment(algorithms, problems, num_runs):
    jobs = []
    max_evaluations = 30000
    population_size = 600

    for run in range(num_runs):
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


def experiment(algorithms=[mlsga, nsgaii, moead, omopso],
               problems=[Srinivas(), ZDT1(), ZDT2(), ZDT6()],
               number_of_runs=30):

    jobs = configure_experiment(algorithms, problems, number_of_runs)

    output_directory = "data"
    experiment = Experiment(output_directory, jobs)
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
    algo.observable.register(observer=ProgressBarObserver(max=max_evaluations))
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

    #generate_boxplot("QualityIndicatorSummary.csv")

    experiment()
    #run_mlsga()
