import math
import numpy
import random
import sympy
import time

from cmlsga.algorithms.heia import HEIA
from cmlsga.mls import MultiLevelSelection

from jmetal.algorithm.multiobjective.ibea import IBEA
from jmetal.algorithm.multiobjective.nsgaii import NSGAII, DynamicNSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.algorithm.multiobjective.spea2 import SPEA2
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover, SBXCrossover
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.selection import RouletteWheelSelection
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.termination_criterion import StoppingByEvaluations

from jmetal.util.archive import BoundedArchive, NonDominatedSolutionsArchive, CrowdingDistanceArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance

from copy import copy

from sympy import symbols, sin, pi, diff, oo, zoo, nan, cos, lambdify


"""
Extended classes for incremental data output. TODO refactor
"""
def incremental_stopping_condition_is_met(algo):
    eval_step = 1000
    if algo.output_path:
        if algo.termination_criterion.evaluations / (algo.output_count * eval_step) > 1:
            algo.output_job(algo.output_count * eval_step)
            algo.output_count += 1

    return algo.termination_criterion.is_met


"""
All base algorithms with incremental experimental setup.
Output results every 1000 evaluations
Save results in a dominance and crowding distance archive

Algorithms:
U-NSGA-III, MOEA/D-DE, HEIA, IBEA, cMLSGA, CMPSO, SMPSO, OMOPSO

TODO: refactor to use singular wrapper instead of extending every algorithm
"""

class IncrementalNSGAIII(NSGAIII):
    def __init__(self, **kwargs):
        super(IncrementalNSGAIII, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def get_observable_data(self):
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.get_result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time,
            "COUNTER": int(self.evaluations / self.population_size)
        }

    def step(self):
        super(IncrementalNSGAIII, self).step()

        for s in self.solutions:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

    def get_name(self):
        return "U-NSGA-III"



class IncrementalHEIA(HEIA):
    def __init__(self, **kwargs):
        super(IncrementalHEIA, self).__init__(**kwargs)

        #self.solutions_archive = CrowdingDistanceArchive(self.population_size)
        self.output_count = 1

    def get_observable_data(self):
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.get_result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time,
            "COUNTER": int(self.evaluations / self.population_size)
        }
    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)


class NSGAIIe(IncrementalNSGAIII):
    def __init__(self, **kwargs):
        super(NSGAIIe, self).__init__(**kwargs)
        self.epigenetic_proba = 0.1
        self.block_size = int(self.problem.number_of_variables / 5)


    def reproduction(self, mating_population):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
            raise Exception("Wrong number of parents")

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

            epigenetic_mask = []
            epigenetic_parent = parents[0]
            p1_has_mask = hasattr(parents[0], "epigenetic_mask")
            p2_has_mask = hasattr(parents[1], "epigenetic_mask")

            if p1_has_mask and p2_has_mask:
                all_mask = set(parents[0].epigenetic_mask + parents[1].epigenetic_mask)
                epigenetic_mask = random.sample(all_mask, self.block_size)
            elif p1_has_mask:
                epigenetic_mask = parents[0].epigenetic_mask
            elif p2_has_mask:
                epigenetic_mask = parents[1].epigenetic_mask
                epigenetic_parent = parents[1]

            for solution in offspring:
                if random.random() < self.epigenetic_proba:
                    # non-contiguous block
                    if len(epigenetic_mask) == 0:
                        epigenetic_mask = random.sample(range(len(solution.variables)), min(len(solution.variables), self.block_size))

                    # contiguous block
                    #if len(epigenetic_mask) == 0:
                    #    block_start = random.randint(0, solution.number_of_variables - self.block_size)
                    #    epigenetic_mask = [i for i in range(block_start, block_start + self.block_size)]

                    for v in epigenetic_mask:
                        solution.variables[v] = epigenetic_parent.variables[v]

                    if len(epigenetic_mask) != 0: 
                        solution.epigenetic_mask = epigenetic_mask

                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def get_name(self):
        return "U-NSGA-IIIe-non-cont"


class IncrementalcMLSGA(MultiLevelSelection):
    def __init__(self, **kwargs):
        super(IncrementalcMLSGA, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

    def step(self):
        super(IncrementalcMLSGA, self).step()

        for s in self.solutions:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list


class IncrementalHEIA(HEIA):
    def __init__(self, **kwargs):
        super(IncrementalHEIA, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

    def step(self):
        super(IncrementalHEIA, self).step()

        for s in self.solutions:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list



class IncrementalIBEA(IBEA):
    def __init__(self, **kwargs):
        super(IncrementalIBEA, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

    def step(self):
        super(IncrementalIBEA, self).step()

        for s in self.solutions:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }



class IncrementalMOEAD(MOEAD):
    def __init__(self, **kwargs):
        super(IncrementalMOEAD, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)


    def update_progress(self):
        if (hasattr(self.problem, 'the_problem_has_changed') and
            self.problem.the_problem_has_changed()):

            self.solutions = self.evaluate(self.solutions)
            self.problem.clear_changed()

        self.evaluations += self.offspring_population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def step(self):
        super(IncrementalMOEAD, self).step()

        for s in self.solutions:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list

    def get_observable_data(self):
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.get_result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time,
            "COUNTER": int(self.evaluations / self.population_size)
        }

    def get_name(self):
        return "MOEAD"



class MOEADe(IncrementalMOEAD):
    def __init__(self, **kwargs): #epigenetic_proba, block_size, **kwargs):
        super(MOEADe, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1 #epigenetic_proba
        self.block_size = int(self.problem.number_of_variables / 5) #block_size

    def get_observable_data(self):
        return {
                "PROBLEM": self.problem,
                "EVALUATIONS": self.evaluations,
                "SOLUTIONS": self.get_result(),
                "ALL_SOLUTIONS": self.solutions,
                "COUNTER": int(self.evaluations / self.population_size)
            }

    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]
        offspring_population = self.crossover_operator.execute(mating_population)
        offspring = offspring_population[0]

        #print(offspring.variables)

        # blocking
        #block_max = offspring.number_of_variables / 2
        #block_size = max(2, int((self.evaluations / 100000.0) * block_max))

        #proba_max = 0.8
        #epigenetic_proba = max(0.1, (self.evaluations / 100000.0) * proba_max)

        epigenetic_mask = []
        epigenetic_parent = mating_population[0]
        p1_has_mask = hasattr(mating_population[0], "epigenetic_mask")
        p2_has_mask = hasattr(mating_population[1], "epigenetic_mask")

        if p1_has_mask and p2_has_mask:
            all_mask = set(mating_population[0].epigenetic_mask + mating_population[1].epigenetic_mask)
            epigenetic_mask = random.sample(all_mask, self.block_size)
        elif p1_has_mask:
            epigenetic_mask = mating_population[0].epigenetic_mask
        elif p2_has_mask:
            epigenetic_mask = mating_population[1].epigenetic_mask
            epigenetic_parent = mating_population[1]


        if random.random() < self.epigenetic_proba:
            # non-contiguous block
            #if len(epigenetic_mask) == 0:
            #    epigenetic_mask = random.sample(range(len(offspring.variables)), min(len(offspring.variables), self.block_size))

            # contiguous block
            if len(epigenetic_mask) == 0:
                block_start = random.randint(0, offspring.number_of_variables - self.block_size)
                epigenetic_mask = [i for i in range(block_start, block_start + self.block_size)]

            for v in epigenetic_mask:
                offspring.variables[v] = epigenetic_parent.variables[v]

            # contiguous block
            #block = self.block_size
            #block_start = random.randint(0, offspring.number_of_variables - block)
            #for v in range(block_start, block_start + block):
            #    offspring.variables[v] = mating_population[0].variables[v]

        self.mutation_operator.execute(offspring)

        if len(epigenetic_mask) != 0: 
            offspring.epigenetic_mask = epigenetic_mask

        return offspring_population


    def get_name(self):
        return "MOEAD-e-cont"


class MOEADeGP1(MOEADe):
    def __init__(self, **kwargs):
        super(MOEADeGP1, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        #self.xvars = symbols("x0:{}".format(self.problem.number_of_variables))


    def get_differentials(self):
        return self.problem.differentials(self)

    def get_gradients(self, values):
        diffs = self.problem.differentials()
        xvars = self.problem.xvars
        gradients = []
        for j in range(self.problem.number_of_variables):
            f0 = diffs[0][j].subs(
                [(xvars[i],values[i]) for i in range(self.problem.number_of_variables)]
            )
            f1 = diffs[1][j].subs(
	        [(xvars[i],values[i]) for i in range(self.problem.number_of_variables)]
            )

            try:
                g0 = sympy.N(f0)
                g1 = sympy.N(f1)
            except ValueError:
                g0 = 0
                g1 = 0

            #f0 = differentials[0].diff(self.xvars[j]).subs(
            #        [(self.xvars[i], v) for (i,v) in enumerate(values)]
            #    ).evalf()

            #f1 = differentials[1].diff(self.xvars[j]).subs(
            #        [(self.xvars[i], v) for (i,v) in enumerate(values)]
            #    ).evalf()

            gradients.append(g0 + g1)

        return gradients


    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

        for offspring in offspring_population:
            if (random.random() < self.epigenetic_proba):
                gradients = self.get_gradients(offspring.variables)

                # choose block based on gradient
                for (i,g) in enumerate(gradients):
                    try:
                        if (g > 0):
                            offspring.variables[i] = mating_population[0].variables[i]
                    except Exception as e:
                        print(e)


        self.mutation_operator.execute(offspring_population[0])

        return offspring_population


    def get_name(self):
        return "MOEAD-eGP1"


class MOEADeGN1(MOEADeGP1):
    def __init__(self, **kwargs):
        super(MOEADeGN1, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

        
        for offspring in offspring_population:
            if (random.random() < self.epigenetic_proba):
                gradients = self.get_gradients(offspring.variables)

                # choose block based on gradient
                for (i,g) in enumerate(gradients):
                    try:
                        if (g < 0):
                            offspring.variables[i] = mating_population[0].variables[i]
                    except Exception as e:
                        print(e)

            self.mutation_operator.execute(offspring)

        return offspring_population

    def get_name(self):
        return "MOEAD-eGN1"


# gradient positive (more) probability
class MOEADegpp(MOEADeGP1):
    def __init__(self, **kwargs):
        super(MOEADegpp, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

        diffs = self.get_differentials()
        for offspring in offspring_population:
            gradients = self.get_gradients(diffs, offspring.variables)

            # choose block based on gradient
            for (i,g) in enumerate(gradients):
                if not g.has(oo, -oo, nan, zoo) and g > 0:
                    self.epigenetic_proba = 0.5
                if random.random() < self.epigenetic_proba:
                    offspring.variables[i] = mating_population[0].variables[i]
                self.epigenetic_proba = 0.1

            self.mutation_operator.execute(offspring)

        return offspring_population

    def get_name(self):
        return "MOEAD-egpp"

# gradient negative (less) probability
class MOEADegnp(MOEADeGP1):
    def __init__(self, **kwargs):
        super(MOEADegnp, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

        diffs = self.get_differentials()
        for offspring in offspring_population:
            gradients = self.get_gradients(diffs, offspring.variables)

            # choose block based on gradient
            for (i,g) in enumerate(gradients):
                if not g.has(oo, -oo, nan, zoo) and g < 0:
                    self.epigenetic_proba = 0.5
                if random.random() < self.epigenetic_proba:
                    offspring.variables[i] = mating_population[0].variables[i]
                self.epigenetic_proba = 0.1

            self.mutation_operator.execute(offspring)

        return offspring_population

    def get_name(self):
        return "MOEAD-egnp"


class MOEADeip(MOEADe):
    def __init__(self, **kwargs):
        super(MOEADeip, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        proba_max = 0.8
        self.epigenetic_proba = max(0.1, (self.evaluations / 100000.0) * proba_max)

        return super(MOEADeip, self).reproduction(mating_population)

    def get_name(self):
        return "MOEAD-e-ip"


class MOEADeib(MOEADe):
    def __init__(self, **kwargs):
        super(MOEADeib, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        block_max = mating_population[0].number_of_variables / 2
        self.block_size = max(2, int((self.evaluations / 100000.0) * block_max))

        return super(MOEADeib, self).reproduction(mating_population)

    def get_name(self):
        return "MOEAD-e-ib"


class NSGAIIeib(NSGAIIe):
    def __init__(self, **kwargs):
        super(NSGAIIeib, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        block_max = mating_population[0].number_of_variables / 2
        self.block_size = max(2, int((self.evaluations / 100000.0) * block_max))

        return super(NSGAIIeib, self).reproduction(mating_population)

    def get_name(self):
        return "NSGAII-e-ib"


class NSGAIIeip(NSGAIIe):
    def __init__(self, **kwargs):
        super(NSGAIIeip, self).__init__(**kwargs)

    def reproduction(self, mating_population):
        proba_max = 0.8
        self.epigenetic_proba = max(0.1, (self.evaluations / 100000.0) * proba_max)

        return super(NSGAIIeip, self).reproduction(mating_population)

    def get_name(self):
        return "NSGAII-e-ip"



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

    def get_name(self):
        return "MOGA"


class MOGeneticAlgorithme(MOGeneticAlgorithm):
    def __init__(self, **kwargs):
        super(MOGeneticAlgorithme, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        self.block_size = 6


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
                if random.random() < self.epigenetic_proba:
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
        return "MOGAe"



def random_search(problem, population_size, max_evaluations, evaluator):
    return (
        RandomSearch,
        {
            "problem": problem,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )


def moga(problem, population_size, max_evaluations, evaluator):
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

def mogae(problem, population_size, max_evaluations, evaluator):
    return (
        MOGeneticAlgorithme,
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


def cmlsga(algorithms, problem, population_size, max_evaluations, evaluator, number_of_collectives=8, reproduction_delay=8):
    return (
        IncrementalcMLSGA,
        {
            "problem": problem,
            "population_size": population_size,
            "max_evaluations": max_evaluations,
            "number_of_collectives": 8,
            "reproduction_delay": 8,
            "algorithms": algorithms,
            "termination_criterion": StoppingByEvaluations(max_evaluations)
        }
    )


def nsgaii(problem, population_size, max_evaluations, evaluator): #, mutation_rate=0.1, crossover_rate=1.0):
    return (
        IncrementalNSGAII,
        {
            "population_evaluator": evaluator
        }
    )


def nsgaiieip(problem, population_size, max_evaluations, evaluator, m=0.1, c=1.0):
    return (
        NSGAIIeip,
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


def unsgaiii(problem, population_size, max_evaluations, evaluator, mutation_rate=-1.0, crossover_rate=0.9, n_points=16):
    if mutation_rate == -1.0:
        mutation_rate = 1.0 / problem.number_of_variables

    return (
        IncrementalNSGAIII,
        {
            "problem": problem,
            "population_size": population_size,
            "reference_directions": UniformReferenceDirectionFactory(
                problem.number_of_objectives,
                n_points=n_points
            ),
            "mutation": PolynomialMutation(
                mutation_rate,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=crossover_rate, distribution_index=30),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        }
    )

def unsgaiiie(problem, population_size, max_evaluations, evaluator, mutation_rate=-1.0, crossover_rate=0.9, n_points=16):
    if mutation_rate == -1.0:
        mutation_rate = 1.0 / problem.number_of_variables

    return (
        NSGAIIe,
        {
            "problem": problem,
            "population_size": population_size,
            "reference_directions": UniformReferenceDirectionFactory(
                problem.number_of_objectives,
                n_points=n_points
            ),
            "mutation": PolynomialMutation(
                1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "crossover": SBXCrossover(probability=crossover_rate, distribution_index=30),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        }
    )



def ibea(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate, kappa=0.05):
    return (
        IncrementalIBEA,
        {
            "problem": problem,
            "kappa": kappa,
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

def moead_options(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0, F=0.5, neighbour_size=3, neighbour_selection=0.9):
    return {
        "problem": problem,
        "population_size": population_size,
        "crossover": DifferentialEvolutionCrossover(CR=crossover_rate, F=F, K=0.5), #
        "mutation": PolynomialMutation(
            mutation_rate,
            distribution_index=20
        ),
        "aggregative_function": Tschebycheff(dimension=problem.number_of_objectives),
        "neighbor_size": neighbour_size,
        "neighbourhood_selection_probability": neighbour_selection,
        "max_number_of_replaced_solutions": 2,
        "weight_files_path": "resources/MOEAD_weights",
        "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        "population_evaluator": evaluator
    }

def moead(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0,F=0.5, neighbour_size=3, neighbour_selection=0.9):
    return (
        IncrementalMOEAD,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, mutation_rate, crossover_rate,
                      F, neighbour_size, neighbour_selection)
        )
#def moead(problem, population_size, max_evaluations, evaluator):
#    return (
#        MOEAD,
#        moead_options(problem, population_size, max_evaluations,
#                      evaluator, mutation_rate, crossover_rate)
#        )
def moead(problem, population_size, max_evaluations, evaluator):
    return (
        IncrementalMOEAD,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )

def moeade(problem, population_size, max_evaluations, evaluator, 
    m=0.1, c=1.0,F=0.5,nsize=3,nselect=0.9):
    return (
        MOEADe,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, m, c, F, nsize, nselect)
    )

def moeadeip(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0):
    return (
        MOEADeip,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, mutation_rate, crossover_rate)
    )

def moeadeib(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0):
    return (
        MOEADeib,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, mutation_rate, crossover_rate)
    )

def moeadegp1(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADeGP1,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )

def moeadegn1(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADeGN1,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )


def moeadegpp(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADegpp,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )
def moeadegnp(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADegnp,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )


def heia(problem, population_size, max_evaluations, evaluator, m=-1.0, c=0.7, NA=20, theta=0.9):
    if m == -1.0:
        m = 1.0 / problem.number_of_variables

    return (
        IncrementalHEIA,
        {
            "problem": problem,
            "population_size": population_size,
            "mutation": PolynomialMutation(
                probability= m,
                distribution_index=20
            ),
            "crossover": SBXCrossover(c),
            "selection": None,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "population_evaluator": evaluator,
            "NA": NA,
            "theta": theta
        }
    )
            


