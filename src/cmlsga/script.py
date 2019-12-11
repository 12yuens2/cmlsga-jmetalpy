from jmetal.config import store
from jmetal.core.quality_indicator import *
from jmetal.lab.visualization import Plot
from jmetal.util.solutions import read_solutions


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

    generate_boxplot("QualityIndicatorSummary.csv")

    #experiment()
    #run_mlsga()
