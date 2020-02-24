from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.termination_criterion import StoppingByEvaluations

from cmlsga.mls import MultiLevelSelection

class MOGeneticAlgorithm(GeneticAlgorithm):
    def __init__( self, problem, population_size, offspring_population_size,
                  mutation, crossover, selection,
                  termination_criterion=store.default_termination_criteria,
                  population_generator=store.default_generator,
                  population_evaluator=store.default_evaluator):

        super(MOGeneticAlgorithm, self).__init__(
            problem, population_size, offspring_population_size,
            mutation, crossover, selection)

    def replacement(self, population, offspring_population):
        population.extend(offspring_population)

        # average fitness of all objectives (MLS1)
        population.sort(key=lambda s: sum([o for o in s.objectives])/len(s.objectives))

        return population[:self.population_size]

    def get_result(self):
        return self.solutions


def random_search(problem, population_size, max_evaluations, evaluator):
    return (
        RandomSearch,
        {
            "problem": problem,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )


def genetic_algorithm(problem, population_size, max_evaluations, evaluator):
    return (
        MOGeneticAlgorithm,
        {
            "problem": problem,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(0.08, 20),
            "crossover": SBXCrossover(0.7),
            "selection": RouletteWheelSelection(),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )


def mlsga(algorithms, problem, population_size, max_evaluations, evaluator):
    return (
        MultiLevelSelection,
        {
            "problem": problem,
            "population_size": population_size,
            "max_evaluations": max_evaluations,
            "number_of_collectives": 6,
            "algorithms": algorithms
        }
    )


def nsgaii(problem, population_size, max_evaluations, evaluator):
    return (
        NSGAII,
        {
            "problem": problem,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(
                1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "population_evaluator": evaluator
        }
    )


def spea2(problem, population_size, max_evaluations, evaluator):
    return (
        SPEA2,
        {
            "problem": problem,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "termination_criterion":StoppingByEvaluations(max_evaluations=max_evaluations)
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
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "population_evaluator": evaluator
        }
    )
