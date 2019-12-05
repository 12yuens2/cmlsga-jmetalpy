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

def run_mlsga():
    problem = ZDT6()
    problem.reference_front = read_solutions(filename="resources/reference_front/ZDT6.pf")

    max_evaluations = 30000

    mls = MultiLevelSelection(
        number_of_collectives=6,
        max_evaluations=max_evaluations,
        problem=problem,
        algorithms=[genetic_algorithm]
    )
    mls.run()

    print("nsga2")
    nsga, kwargs = nsgaii(problem, 600, 30000, store.default_evaluator)
    nsga2 = nsga(**kwargs)
    nsga2.run()

    print("moead")
    moea, kwargs = moead(problem, 600, 30000, store.default_evaluator)
    moea_d = moea(**kwargs)
    moea_d.run()

    print("omopso")
    mopso, kwargs = omopso(problem, 600, 30000, store.default_evaluator)
    ompso = mopso(**kwargs)
    ompso.run()

    front = mls.get_result()
    nsga2_front = nsga2.get_result()
    moead_front = moea_d.get_result()
    omopso_front = ompso.get_result()

    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(
        [front, nsga2_front, moead_front, omopso_front],
        label=["MLS", "NSGAII", "MOEAD", "OMOPSO"], filename="MLS")

    return 0


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
