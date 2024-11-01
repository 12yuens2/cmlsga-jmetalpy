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

<<<<<<< HEAD
from jmetal.util.archive import BoundedArchive, NonDominatedSolutionsArchive, CrowdingDistanceArchive
=======
from jmetal.util.archive import BoundedArchive
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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


<<<<<<< HEAD
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
=======
class IncrementalNSGAII(NSGAII):
    def __init__(self, **kwargs):
        super(IncrementalNSGAII, self).__init__(**kwargs)

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        self.output_count = 1

    def get_observable_data(self):
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.get_result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time,
            "COUNTER": int(self.evaluations / self.population_size)
        }

<<<<<<< HEAD
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


=======
    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)


class NSGAIIe(IncrementalNSGAII):
    def __init__(self, **kwargs):
        super(NSGAIIe, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        self.block_size = 6

    
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    def reproduction(self, mating_population):
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        if len(mating_population) % number_of_parents_to_combine != 0:
<<<<<<< HEAD
            raise Exception("Wrong number of parents")
=======
            raise Exception('Wrong number of parents')
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

        offspring_population = []
        for i in range(0, self.offspring_population_size, number_of_parents_to_combine):
            parents = []
            for j in range(number_of_parents_to_combine):
                parents.append(mating_population[i + j])

            offspring = self.crossover_operator.execute(parents)

<<<<<<< HEAD
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
=======
            for solution in offspring:

                #blocking
                if random.random() < self.epigenetic_proba:
                    block = int(solution.number_of_variables / self.block_size)
                    block_start = random.randint(0, solution.number_of_variables - block)
                    for v in range(block_start, block_start + block):
                        solution.variables[v] = parents[0].variables[v]
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

                self.mutation_operator.execute(solution)
                offspring_population.append(solution)

                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

<<<<<<< HEAD
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


=======

    def get_name(self):
        return "NSGA-IIe"


class IncrementalNSGAIII(NSGAIII):
    def __init__(self, **kwargs):
        super(IncrementalNSGAIII, self).__init__(**kwargs)


    def get_name(self):
        return "U-NSGA-III"

    #    self.output_count = 1


    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

class IncrementalIBEA(IBEA):
    def __init__(self, **kwargs):
        super(IncrementalIBEA, self).__init__(**kwargs)

<<<<<<< HEAD
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
=======
    def get_name(self):
        return "IBEA"
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }

<<<<<<< HEAD
=======
    #    self.output_count = 1

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)

class IncrementalcMLSGA(MultiLevelSelection):
    def __init__(self, **kwargs):
        super(IncrementalcMLSGA, self).__init__(**kwargs)
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b


class IncrementalMOEAD(MOEAD):
    def __init__(self, **kwargs):
        super(IncrementalMOEAD, self).__init__(**kwargs)

<<<<<<< HEAD
        self.solutions_archive = NonDominatedSolutionsArchive()
=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)


    def update_progress(self):
        if (hasattr(self.problem, 'the_problem_has_changed') and
            self.problem.the_problem_has_changed()):

            self.solutions = self.evaluate(self.solutions)
            self.problem.clear_changed()

<<<<<<< HEAD
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
=======
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

        self.evaluations += self.offspring_population_size
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    def get_observable_data(self):
        return {
            'PROBLEM': self.problem,
            'EVALUATIONS': self.evaluations,
            'SOLUTIONS': self.get_result(),
            'COMPUTING_TIME': time.time() - self.start_computing_time,
            "COUNTER": int(self.evaluations / self.population_size)
        }

<<<<<<< HEAD
    def get_name(self):
        return "MOEAD"



class MOEADe(IncrementalMOEAD):
    def __init__(self, **kwargs): #epigenetic_proba, block_size, **kwargs):
        super(MOEADe, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1 #epigenetic_proba
        self.block_size = int(self.problem.number_of_variables / 5) #block_size
=======


class MOEADe(IncrementalMOEAD):
    def __init__(self, **kwargs):
        super(MOEADe, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        self.block_size = 6
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

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
<<<<<<< HEAD
=======

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        offspring_population = self.crossover_operator.execute(mating_population)
        offspring = offspring_population[0]

        #print(offspring.variables)

        # blocking
<<<<<<< HEAD
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

=======
        block_max = offspring.number_of_variables / 2
        block_size = max(2, int((self.evaluations / 100000.0) * block_max))

        proba_max = 0.8
        epigenetic_proba = max(0.1, (self.evaluations / 100000.0) * proba_max)

        if random.random() < self.epigenetic_proba:
            block = int(offspring.number_of_variables / self.block_size)
            block_start = random.randint(0, offspring.number_of_variables - block)
            for v in range(block_start, block_start + block):
                offspring.variables[v] = mating_population[0].variables[v]

        self.mutation_operator.execute(offspring)

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        return offspring_population


    def get_name(self):
<<<<<<< HEAD
        return "MOEAD-e-cont"


class MOEADeGP1(MOEADe):
    def __init__(self, **kwargs):
        super(MOEADeGP1, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        #self.xvars = symbols("x0:{}".format(self.problem.number_of_variables))
=======
        return "MOEAD-e"


class MOEADegy(MOEADe):
    def __init__(self, **kwargs):
        super(MOEADegy, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1
        self.xvars = symbols("x0:{}".format(self.problem.number_of_variables))
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b


    def get_differentials(self):
        return self.problem.differentials(self)

<<<<<<< HEAD
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
=======
    def get_gradients(self, differentials, values):
        #nT = 10
#        maxf = (1 / nT + 2 * 0.1) * (math.sin(2 * nT * math.pi * values[0]) - abs(2 * nT * self.problem.gt))
#
#        if maxf < 0:
#            maxf = 0

        gradients = []
        for j in range(self.problem.number_of_variables):
            f0 = lambdify(self.xvars, differentials[0].diff(self.xvars[j]), modules=[sympy])
            f1 = lambdify(self.xvars, differentials[1].diff(self.xvars[j]), modules=[sympy])

            try:
                g0 = f0(*values)
                g1 = f1(*values)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            except ValueError:
                g0 = 0
                g1 = 0

            #f0 = differentials[0].diff(self.xvars[j]).subs(
            #        [(self.xvars[i], v) for (i,v) in enumerate(values)]
            #    ).evalf()

            #f1 = differentials[1].diff(self.xvars[j]).subs(
            #        [(self.xvars[i], v) for (i,v) in enumerate(values)]
            #    ).evalf()

<<<<<<< HEAD
            gradients.append(g0 + g1)
=======
            gradients.append(sympy.N(g0 + g1))
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

        return gradients


    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

<<<<<<< HEAD
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

=======
        diffs = self.get_differentials()
        for offspring in offspring_population:
            gradients = self.get_gradients(diffs, offspring.variables)

            # choose block based on gradient
            for (i,g) in enumerate(gradients):
                if (not g.has(oo, -oo, nan, zoo)
                    and g > 0
                    and (random.random() < self.epigenetic_proba)):

                    offspring.variables[i] = mating_population[0].variables[i]
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

        self.mutation_operator.execute(offspring_population[0])

        return offspring_population


    def get_name(self):
<<<<<<< HEAD
        return "MOEAD-eGP1"


class MOEADeGN1(MOEADeGP1):
    def __init__(self, **kwargs):
        super(MOEADeGN1, self).__init__(**kwargs)
=======
        return "MOEAD-egy"


class MOEADegn(MOEADegy):
    def __init__(self, **kwargs):
        super(MOEADegn, self).__init__(**kwargs)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    def reproduction(self, mating_population):
        self.crossover_operator.current_individual = self.solutions[self.current_subproblem]

        offspring_population = self.crossover_operator.execute(mating_population)

<<<<<<< HEAD
        
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
=======
        diffs = self.get_differentials()
        for offspring in offspring_population:
            gradients = self.get_gradients(diffs, offspring.variables)

            # choose block based on gradient
            for (i,g) in enumerate(gradients):
                if not g.has(oo, -oo, nan, zoo) and g < 0 and (random.random() < self.epigenetic_proba):
                    offspring.variables[i] = mating_population[0].variables[i]
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

            self.mutation_operator.execute(offspring)

        return offspring_population

    def get_name(self):
<<<<<<< HEAD
        return "MOEAD-eGN1"


# gradient positive (more) probability
class MOEADegpp(MOEADeGP1):
=======
        return "MOEAD-egn"


# gradient positive (more) probability
class MOEADegpp(MOEADegy):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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
<<<<<<< HEAD
class MOEADegnp(MOEADeGP1):
=======
class MOEADegnp(MOEADegy):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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

<<<<<<< HEAD

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


=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

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
<<<<<<< HEAD
        IncrementalcMLSGA,
=======
        #IncrementalcMLSGA,
        MultiLevelSelection,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        {
            "problem": problem,
            "population_size": population_size,
            "max_evaluations": max_evaluations,
            "number_of_collectives": 8,
<<<<<<< HEAD
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
=======
            "algorithms": algorithms,
            "termination_criterion": StoppingByEvaluations(max_evaluations)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        }
    )


<<<<<<< HEAD
def nsgaiieip(problem, population_size, max_evaluations, evaluator, m=0.1, c=1.0):
=======
def nsgaii(problem, population_size, max_evaluations, evaluator): #, mutation_rate=0.1, crossover_rate=1.0):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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
<<<<<<< HEAD
=======
            "population_evaluator": evaluator
        }
    )

def nsgaiie(problem, population_size, max_evaluations, evaluator):
    return (
        NSGAIIe,
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
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            "population_evaluator": evaluator
        }
    )


<<<<<<< HEAD
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
=======
def nsgaiii(problem, population_size, max_evaluations, evaluator): #, mutation_rate, crossover_rate):
    return (
        NSGAIII,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        {
            "problem": problem,
            "population_size": population_size,
            "reference_directions": UniformReferenceDirectionFactory(
                problem.number_of_objectives,
<<<<<<< HEAD
                n_points=n_points
=======
                n_points=population_size-1
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            ),
            "mutation": PolynomialMutation(
                1.0 / problem.number_of_variables,
                distribution_index=20
            ),
<<<<<<< HEAD
            "crossover": SBXCrossover(probability=crossover_rate, distribution_index=30),
=======
            "crossover": SBXCrossover(probability=1.0, distribution_index=20),
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        }
    )


<<<<<<< HEAD

def ibea(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate, kappa=0.05):
    return (
        IncrementalIBEA,
        {
            "problem": problem,
            "kappa": kappa,
=======
def ibea(problem, population_size, max_evaluations, evaluator): #, mutation_rate, crossover_rate):
    return (
        IBEA,
        {
            "problem": problem,
            "kappa": 1,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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

<<<<<<< HEAD
def moead_options(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0, F=0.5, neighbour_size=3, neighbour_selection=0.9):
    return {
        "problem": problem,
        "population_size": population_size,
        "crossover": DifferentialEvolutionCrossover(CR=crossover_rate, F=F, K=0.5), #
=======
def moead_options(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate):
    return {
        "problem": problem,
        "population_size": population_size,
        "crossover": DifferentialEvolutionCrossover(CR=crossover_rate, F=0.5, K=0.5),
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        "mutation": PolynomialMutation(
            mutation_rate,
            distribution_index=20
        ),
        "aggregative_function": Tschebycheff(dimension=problem.number_of_objectives),
<<<<<<< HEAD
        "neighbor_size": neighbour_size,
        "neighbourhood_selection_probability": neighbour_selection,
=======
        "neighbor_size": 3,
        "neighbourhood_selection_probability": 0.9,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        "max_number_of_replaced_solutions": 2,
        "weight_files_path": "resources/MOEAD_weights",
        "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
        "population_evaluator": evaluator
    }

<<<<<<< HEAD
def moead(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0,F=0.5, neighbour_size=3, neighbour_selection=0.9):
    return (
        IncrementalMOEAD,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, mutation_rate, crossover_rate,
                      F, neighbour_size, neighbour_selection)
        )
#def moead(problem, population_size, max_evaluations, evaluator):
=======
#def moead(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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

<<<<<<< HEAD
def moeade(problem, population_size, max_evaluations, evaluator, 
    m=0.1, c=1.0,F=0.5,nsize=3,nselect=0.9):
    return (
        MOEADe,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, m, c, F, nsize, nselect)
    )

def moeadeip(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0):
=======
def moeade(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADe,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )

def moeadeip(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    return (
        MOEADeip,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, mutation_rate, crossover_rate)
    )

<<<<<<< HEAD
def moeadeib(problem, population_size, max_evaluations, evaluator, mutation_rate=0.1, crossover_rate=1.0):
=======
def moeadeib(problem, population_size, max_evaluations, evaluator, mutation_rate, crossover_rate):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    return (
        MOEADeib,
        moead_options(problem, population_size, max_evaluations,
                      evaluator, mutation_rate, crossover_rate)
    )

<<<<<<< HEAD
def moeadegp1(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADeGP1,
=======
def moeadegy(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADegy,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        moead_options(problem, population_size, max_evaluations,
                      evaluator, 1 / problem.number_of_variables, 1.0)
    )

<<<<<<< HEAD
def moeadegn1(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADeGN1,
=======
def moeadegn(problem, population_size, max_evaluations, evaluator):
    return (
        MOEADegn,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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


<<<<<<< HEAD
def heia(problem, population_size, max_evaluations, evaluator, m=-1.0, c=0.7, NA=20, theta=0.9):
    if m == -1.0:
        m = 1.0 / problem.number_of_variables

    return (
        IncrementalHEIA,
=======
def heia(problem, population_size, max_evaluations, evaluator):
    return (
        HEIA,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        {
            "problem": problem,
            "population_size": population_size,
            "mutation": PolynomialMutation(
                probability= m,
                distribution_index=20
            ),
<<<<<<< HEAD
            "crossover": SBXCrossover(c),
            "selection": None,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "population_evaluator": evaluator,
            "NA": NA,
            "theta": theta
=======
            "crossover": SBXCrossover(0.7),
            "selection": None,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "population_evaluator": evaluator
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        }
    )
            


