from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator import PolynomialMutation, UniformMutation, DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.mutation import NonUniformMutation
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations

from mls import MultiLevelSelection

def nsgaii(problem, population_size, max_evaluations):
    return (
        NSGAII,
        { "problem": problem,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations)
        }
    )

def moead(problem, population_size, max_evaluations):
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
            "termination_criterion": StoppingByEvaluations(max=max_evaluations)
        }
    )

def omopso(problem, population_size, max_evaluations):
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
                perturbation=0.5
            ),
            "leaders": CrowdingDistanceArchive(100),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations)
        }
    )


if __name__ == "__main__":
    problem = ZDT1()
    problem.reference_front = "" #read_solutions(filename="resources/reference_front/ZDT1.pf")

    max_evaluations = 30000

    mls = MultiLevelSelection(
        problem=problem,
        algorithms=[nsgaii, moead, omopso]
    )
    mls.run()
