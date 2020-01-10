from functools import partial

from jmetal.problem.multiobjective.zdt import *
from jmetal.problem.multiobjective.lz09 import *
from jmetal.lab.visualization import Plot
from jmetal.util.solutions import read_solutions
from jmetal.util.observer import ProgressBarObserver

from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *


def run_algorithm(algorithm, problem=ZDT1(), population_size=600,
                  max_evaluations=10000, evaluator=store.default_evaluator):

    constructor, kwargs = algorithm(problem, population_size, max_evaluations, evaluator)
    algorithm = constructor(**kwargs)
    algorithm.observable.register(ProgressBarObserver(max=max_evaluations))
    algorithm.run()

    return algorithm.get_result()

reference_front = read_solutions(filename="resources/reference_front/ZDT1.pf")
nsgaii_front = run_algorithm(nsgaii)
moead_front = run_algorithm(moead)
omopso_front = run_algorithm(omopso)
mlsga_front = run_algorithm(partial(mlsga, [nsgaii]))

plot_front = Plot(reference_front=reference_front, axis_labels=["x", "y"])
plot_front.plot([nsgaii_front, moead_front, omopso_front, mlsga_front],
                label=["NSGAII", "MOEAD", "OMOPSO", "MLSGA"])

