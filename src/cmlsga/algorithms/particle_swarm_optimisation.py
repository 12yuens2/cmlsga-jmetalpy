import numpy as np
import random

from copy import copy

from cmlsga.algorithms.cmpso import CMPSO

from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.operator import PolynomialMutation, UniformMutation
from jmetal.operator.mutation import NonUniformMutation
<<<<<<< HEAD
from jmetal.util.archive import CrowdingDistanceArchive, NonDominatedSolutionsArchive
=======
from jmetal.util.archive import CrowdingDistanceArchive
<<<<<<< HEAD
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations

"""
Extended classes for incremental data output. TODO refactor
"""
def incremental_stopping_condition_is_met(algo):
    if algo.output_path:
<<<<<<< HEAD
<<<<<<< HEAD
        if algo.termination_criterion.evaluations / (algo.output_count * 1000) > 1:
            algo.output_job(algo.output_count * 1000)
=======
        if algo.termination_criterion.evaluations / (algo.output_count * 10000) > 1:
            algo.output_job(algo.output_count * 10000)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
=======
        if algo.termination_criterion.evaluations / (algo.output_count * 10000) > 1:
            algo.output_job(algo.output_count * 10000)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            algo.output_count += 1

    return algo.termination_criterion.is_met

class IncrementalOMOPSO(OMOPSO):
    def __init__(self, **kwargs):
        super(IncrementalOMOPSO, self).__init__(**kwargs)

<<<<<<< HEAD
<<<<<<< HEAD
        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)
=======
=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        self.output_count = 1

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)
<<<<<<< HEAD
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }

<<<<<<< HEAD
    def step(self):
        super(IncrementalOMOPSO, self).step()

        for s in self.epsilon_archive.solution_list:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list


class IncrementalSMPSO(SMPSO):
    def __init__(self, **kwargs):
        super(IncrementalSMPSO, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1
=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }

<<<<<<< HEAD
    def step(self):
        super(IncrementalSMPSO, self).step()

        for s in self.leaders.solution_list:
            self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)


class IncrementalCMPSO(CMPSO):
    def __init__(self, **kwargs):
        super(IncrementalCMPSO, self).__init__(**kwargs)

        self.solutions_archive = NonDominatedSolutionsArchive()
        self.output_count = 1

    def step(self):
        super(IncrementalCMPSO, self).step()

        for s in self.archive:
           self.solutions_archive.add(s)

    def result(self):
        return self.solutions_archive.solution_list

    def get_result(self):
        return self.solutions_archive.solution_list

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)


class OMOPSOe(OMOPSO):
    def __init__(self, **kwargs):
        super(OMOPSOe, self).__init__(**kwargs)

=======

class IncrementalSMPSO(SMPSO):
    def __init__(self, **kwargs):
        super(IncrementalSMPSO, self).__init__(**kwargs)

        self.output_count = 1

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)

class IncrementalCMPSO(CMPSO):
    def __init__(self, **kwargs):
        super(IncrementalCMPSO, self).__init__(**kwargs)

        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

class OMOPSOe(OMOPSO):
    def __init__(self, **kwargs):
        super(OMOPSOe, self).__init__(**kwargs)

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        self.epigenetic_proba = 0.1

=======

class IncrementalSMPSO(SMPSO):
    def __init__(self, **kwargs):
        super(IncrementalSMPSO, self).__init__(**kwargs)

        self.output_count = 1

    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": self.solutions
        }

    #def stopping_condition_is_met(self):
    #    return incremental_stopping_condition_is_met(self)

class IncrementalCMPSO(CMPSO):
    def __init__(self, **kwargs):
        super(IncrementalCMPSO, self).__init__(**kwargs)

        self.output_count = 1

    def stopping_condition_is_met(self):
        return incremental_stopping_condition_is_met(self)

class OMOPSOe(OMOPSO):
    def __init__(self, **kwargs):
        super(OMOPSOe, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    def update_position(self, swarm):
        for i in range(self.swarm_size):
            particle = swarm[i]

            for j in range(particle.number_of_variables):
                if random.random() < 1.0 / particle.number_of_variables:
                    continue
<<<<<<< HEAD

                particle.variables[j] += self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2

=======

                particle.variables[j] += self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    def get_name(self):
        return "OMOPSOe"


class SMPSO_Variant(SMPSO):

    def __init__(self, **kwargs):
        super(SMPSO_Variant, self).__init__(**kwargs)

        self.epigenetic_proba = 0.1


    #def update_velocity(self, swarm):
    #    for i in range(self.swarm_size):
    #        pbest = copy(swarm[i].attributes['local_best'])
    #        gbest = self.select_global_best()
<<<<<<< HEAD
<<<<<<< HEAD

    #        r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
    #        r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
    #        c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
    #        c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

    #        w = round(random.uniform(self.weight_min, self.weight_max), 1)

=======

    #        r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
    #        r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
    #        c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
    #        c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

    #        w = round(random.uniform(self.weight_min, self.weight_max), 1)

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
=======

    #        r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
    #        r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
    #        c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
    #        c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

    #        w = round(random.uniform(self.weight_min, self.weight_max), 1)

>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    #        for var in range(swarm[i].number_of_variables):
    #            swarm_var = swarm[i].variables[var]
    #            self.speed[i][var] = w * self.speed[i][var] \
    #                                   + (c1 * r1 * (pbest.variables[var] - swarm_var)) \
    #                                   + (c2 * r2 * (gbest.variables[var] - swarm_var))

    def update_position(self, swarm) -> None:
        for i in range(self.swarm_size):
            particle = swarm[i]

            for j in range(particle.number_of_variables):
                if random.random() < self.epigenetic_proba:
                    continue
<<<<<<< HEAD
<<<<<<< HEAD

                particle.variables[j] += self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2

    def get_name(self):
        return "SMPSO-e"


def omopso(problem, population_size, max_evaluations, evaluator, uniform_m=-1.0, nonuniform_m=-1.0):
    if uniform_m == -1.0:
        uniform_m = 1.0 / problem.number_of_variables

    if nonuniform_m == -1.0:
        nonuniform_m = 1.0 / problem.number_of_variables

    return (
        IncrementalOMOPSO,
        {
            "problem": problem,
            "swarm_size": 100,
            "epsilon": 0.0075,
            "uniform_mutation": UniformMutation(
                probability=uniform_m,
                perturbation=0.5
            ),
            "non_uniform_mutation": NonUniformMutation(
                probability=nonuniform_m,
                perturbation=0.5,
                max_iterations=int(max_evaluations / population_size)
            ),
            "leaders": CrowdingDistanceArchive(population_size),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "swarm_evaluator": evaluator
        }
    )


def omopsoe(problem, population_size, max_evaluations, evaluator):
    return (
        OMOPSOe,
=======

                particle.variables[j] += self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2

    def get_name(self):
        return "SMPSO-e"


def omopso(problem, population_size, max_evaluations, evaluator): #, m, l):
    return (
        OMOPSO,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "swarm_evaluator": evaluator
        }
    )
=======

                particle.variables[j] += self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    def get_name(self):
        return "SMPSO-e"

<<<<<<< HEAD
<<<<<<< HEAD
def smpso(problem, population_size, max_evaluations, evaluator, m=-1.0):
    if m == -1.0:
        m = 1.0 / problem.number_of_variables
=======
def omopsoe(problem, population_size, max_evaluations, evaluator):
    return (
        OMOPSOe,
=======

def omopso(problem, population_size, max_evaluations, evaluator): #, m, l):
    return (
        OMOPSO,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
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
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "swarm_evaluator": evaluator
        }
    )


<<<<<<< HEAD
def smpso(problem, population_size, max_evaluations, evaluator): #, m, l):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
=======
def omopsoe(problem, population_size, max_evaluations, evaluator):
    return (
        OMOPSOe,
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
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations),
            "swarm_evaluator": evaluator
        }
    )


def smpso(problem, population_size, max_evaluations, evaluator): #, m, l):
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
    return (
        IncrementalSMPSO,
        {
            "problem": problem,
<<<<<<< HEAD
<<<<<<< HEAD
            "swarm_size": 100,
            "mutation": PolynomialMutation(
                probability=m,
                distribution_index=20
            ),
            "leaders": CrowdingDistanceArchive(population_size),
=======
=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            "swarm_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "leaders": CrowdingDistanceArchive(100),
<<<<<<< HEAD
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
=======
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )


def smpsoe(problem, population_size, max_evaluations, evaluator):
    return (
        SMPSO_Variant,
        {
            "problem": problem,
            "swarm_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "leaders": CrowdingDistanceArchive(100),
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )


def cmpso(problem, population_size, max_evaluations, evaluator):
    return (
<<<<<<< HEAD
<<<<<<< HEAD
        IncrementalCMPSO,
=======
        CMPSO,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
=======
        CMPSO,
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b
        {
            "problem": problem,
            "swarm_size": population_size,
            "termination_criterion": StoppingByEvaluations(max_evaluations=max_evaluations)
        }
    )
