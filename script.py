from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.lab.visualization import Plot
from jmetal.operator import PolynomialMutation, UniformMutation, DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import NonUniformMutation
from jmetal.problem.multiobjective.constrained import Srinivas
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.solutions import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

from mls import MultiLevelSelection

def nsgaii(problem, population_size, max_evaluations, evaluator):
    return (
        NSGAII,
        { "problem": problem,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=1.0, distribution_index=5),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations),
            "population_evaluator": evaluator
        }
    )

def moead(problem, population_size, max_evaluations, evaluator):
    return (
        MOEAD,
        {
            "problem": problem,
            "population_size": population_size,
            "crossover": DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "aggregative_function": Tschebycheff(dimension=problem.number_of_objectives),
            "neighbor_size": 3,
            "neighbourhood_selection_probability": 0.9,
            "max_number_of_replaced_solutions": 2,
            "weight_files_path": "resources/MOEAD_weights",
            "termination_criterion": StoppingByEvaluations(max=max_evaluations),
            "population_evaluator": evaluator
        }
    )

def omopso(problem, population_size, max_evaluations, evaluator):
    return (
        OMOPSO,
        {
            "problem": problem,
            "swarm_size": population_size,
            "epsilon": 0.0075,
            "uniform_mutation": UniformMutation(
                probability=1.0 / problem.number_of_variables,
                perturbation=0.5
            ),
            "non_uniform_mutation": NonUniformMutation(
                probability=1.0 / problem.number_of_variables,
                perturbation=0.5,
                max_iterations=int(max_evaluations / population_size)
            ),
            "leaders": CrowdingDistanceArchive(100),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations),
            "swarm_evaluator": evaluator
        }
    )


if __name__ == "__main__":
    problem = Srinivas()
    problem.reference_front = read_solutions(filename="resources/reference_front/Srinivas.pf")

    max_evaluations = 10000

    mls = MultiLevelSelection(
        max_evaluations=max_evaluations,
        problem=problem,
        algorithms=[nsgaii, moead, omopso]
    )
    mls.run()

    #a, b = omopso(problem, 200, max_evaluations)
    #algo = a(**b)
    #algo.run()
    front = mls.get_result()

    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label="MLS", filename="MLS")
