from functools import partial

from jmetal.problem.multiobjective.dtlz import *
from jmetal.problem.multiobjective.zdt import *
from jmetal.problem.multiobjective.lz09 import *
from jmetal.lab.visualization import Plot
from jmetal.util.solutions import read_solutions
from jmetal.util.observer import ProgressBarObserver

from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *


def run_algorithm(algorithm, problem=LZ09_F3(), population_size=600,
                  max_evaluations=100000, evaluator=store.default_evaluator):

    constructor, kwargs = algorithm(problem, population_size, max_evaluations, evaluator)
    algorithm = constructor(**kwargs)
    algorithm.observable.register(ProgressBarObserver(max=max_evaluations))
    algorithm.run()

    return algorithm.get_result(), algorithm.get_name()

def run_plots(algorithms, problems):
    for problem in problems:
        print("Run {}".format(problem.get_name()))

        reference_front = read_solutions(
            filename="resources/reference_front/{}.pf".format(problem.get_name()))

        fronts, labels = zip(*[run_algorithm(algorithm, problem=problem) for algorithm in algorithms])

        plot_front = Plot(plot_title="{} 100000 evaluations".format(problem.get_name()),
                          reference_front=reference_front, axis_labels=["x", "y"])
        plot_front.plot(list(fronts), label=list(labels))


run_plots([nsgaii, moead, omopso, partial(mlsga, [nsgaii])],
          [ZDT1(), DTLZ1()])
