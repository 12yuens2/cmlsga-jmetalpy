import random

from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.util.archive import BoundedArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance

from copy import copy


from cmlsga.mls import MultiLevelSelection

"""
Extended classes for incremental data output. TODO refactor
"""
def incremental_stopping_condition_is_met(algo):
    if algo.output_path:
        if algo.termination_criterion.evaluations / (algo.output_count * 10000) > 1:
            algo.output_job(algo.output_count * 10000)
            algo.output_count += 1

    return algo.termination_criterion.is_met

class NSGAIIe(NSGAII):
    def __init__(self, **kwargs):
        super(NSGAIIe, self).__init__(**kwargs)

        #self.pf = BoundedArchive(1000, DominanceComparator(), CrowdingDistance())

        self.epigenetic_proba = 0.1
        self.block_size = 6

    #def get_observable_data(self):
    #    for solution in self.solutions:
    #        self.pf.add(copy(solution))
    #    return {
    #        "PROBLEM": self.problem,
    #        "EVALUATIONS": self.evaluations,
    #        "SOLUTIONS": self.get_result(),
    #        "ALL_SOLUTIONS": self.pf.solution_list
    #    }

    def reproduction(self, mating_population):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception('Wrong number of parents')

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:

                #blocking
                if random.random < self.epigenetic_proba:
                    block = int(solution.number_of_variables / self.block_size)
                    block_start = random.randint(0, solution.number_of_variables - block)

                    for v in range(block_start, block_start + block):
                        solution.variables[v] = parents[0].variables[v]

                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population


    def get_name(self):
        return "NSGA-IIe"
    #    self.output_count = 1

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)


class IncrementalNSGAIII(NSGAIII):
    def __init__(self, **kwargs):
        super(IncrementalNSGAIII, self).__init__(**kwargs)


    def get_name(self):
        return "U-NSGA-III"

    #    self.output_count = 1


    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)

class IncrementalIBEA(IBEA):
    def __init__(self, **kwargs):
        super(IncrementalIBEA, self).__init__(**kwargs)

    def get_name(self):
        return "IBEA"

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }

    #    self.output_count = 1

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)

class MOEADe(MOEAD):
    def __init__(self, **kwargs):
        super(MOEADe, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        self.block_size = 6

        #self.output_count = 1

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)

    def get_observable_data(self):
        return {
                "PROBLEM": self.problem,
                "EVALUATIONS": self.evaluations,
                "SOLUTIONS": self.get_result(),
                "ALL_SOLUTIONS": self.solutions
            }

    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

        # blocking
        offspring = offspring_population[0]
        if random.random() < self.epigenetic_proba:
            block = int(offspring.number_of_variables / self.block_size)
            block_start = random.randint(0, offspring.number_of_variables - block)
            for v in range(block_start, block_start + block):
                offspring.variables[v] = mating_population[0].variables[v]

        self.mutation_operator.execute(offspring)

        return offspring_population


    def get_name(self):
        return "MOEAD-e"




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
            "number_of_collectives": 8,
            "algorithms": algorithms
        }
    )


def nsgaii(problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalNSGAII,
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

def nsgaiie(problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalNSGAII,
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


def nsgaiii(problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalNSGAIII,
        {
            "problem": problem,
            "population_size": population_size,
            "reference_directions": UniformReferenceDirectionFactory(
                problem.number_of_objectives,
                n_points=population_size-1
            ),
            "mutation": PolynomialMutation(
                1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        }
    )


def ibea(problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalIBEA,
        {
            "problem": problem,
            "kappa": 1,
            "population_size": population_size,
            "offspring_population_size": population_size,
            "mutation": PolynomialMutation(
                1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
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


def moeade(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADe,
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


