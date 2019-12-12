from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation, UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations


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


def smpso(problem, population_size, max_evaluations, evaluator):
    return (
        SMPSO,
        {
            "problem": problem,
            "swarm_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "leaders": CrowdingDistanceArchive(100),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations)
        }
    )
