import numpy as np
import random

from copy import copy

from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.operator import PolynomialMutation, UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import StoppingByEvaluations

class CMPSO(ParticleSwarmOptimization):

    def __init__(self,
                 problem,
                 swarm_size,
                 leaders=CrowdingDistanceArchive(100),
                 termination_criterion=store.default_termination_criteria,
                 swarm_generator = store.default_generator,
                 swarm_evaluator = store.default_evaluator):
        super(CMPSO, self).__init__(problem, swarm_size)
        self.leaders = leaders

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator

        self.dominance_comparator = DominanceComparator()

        # Zhan et al. Adaptive particle swarm optimization 2009
        # Set velocity range as 0.2(upper bound - lower bound)
        self.speed_max = [
            0.2 * (self.problem.upper_bound[i] - self.problem.lower_bound[i])
            for i in range(len(self.problem.upper_bound))
        ]

        self.speed = np.zeros((self.problem.number_of_objectives,
                               self.swarm_size,
                               self.problem.number_of_variables),
                              dtype=float)


    def create_initial_solutions(self):
        return [
            [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]
            for _ in range(self.problem.number_of_objectives)
        ]

    def evaluate(self, swarms):
        return [
            self.swarm_evaluator.evaluate(swarm, self.problem)
            for swarm in swarms
        ]

    def stopping_condition_is_met(self):
        return self.termination_criterion.is_met


    def get_observable_data(self):
        pass


    def init_progress(self):
        super.init_progress()
        self.update_archive()

    def initialize_velocity(self):
        for m in range(self.problem.number_of_objectives):
            for i in range(self.swarm_size):
                for d in range(self.problem.number_of_variables):
                    self.speed[m][i][d] = random.uniform(-self.speed_max, self.speed_max)

        self.solutions = self.evaluate(self.solutions)


    def initialize_particle_best(self, swarms):
        for swarm in swarms:
            for particle in swarm:
                particle.attributes["local_best"] = copy(particle)


    def initialize_global_best(self, swarms):
        self.update_global_best(swarms)


    def update_archive(self):
        pass

    def update_velocity(self, swarms):
        for swarm in swarms:
            for particle in swarm:
                archive_position = self.select_archive_solution()

                self.speed[m][i][d] = \
                w * self.speed[m][i][d]
                + (self.c1 * self.r1[d] * (pbest - position))
                + (self.c2 * self.r2[d] * (gbest - position))
                + (self.c3 * self.r3[d] * (archive - position))


    def update_particle_best(self, swarm):
        for swarm in swarms:
            for particle in swarm:
                flag = self.dominance_comparator.compare(particle,
                                                         particle.attributes["local_best"])
                if flag != 1:
                    particle.attributes["local_host"] = copy(particle)


    def update_global_best(self, swarms):
        for swarm in swarms:
            for particle in swarm:
                self.leaders.add(copy(particle))


    def update_position(self, swarms):
        for m in range(len(swarms)):
            for i in range(len(swarms[m])):
                particle = swarms[m][i]

                for d in range(particle.number_of_variables):
                    particle.variables[d] += self.speed[m][i][d]

                # Limit speed to max velocities
                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= self.change_velocity1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= self.change_velocity2


    def perturbation(self, swarm):
        pass


    def get_result(self):
        pass

    def get_name(self):
        return "CMPSO"



class OMOPSO_Variant(OMOPSO):

    def __init__(self, **kwargs):
        super(OMOPSO_Variant, self).__init__(**kwargs)


    def update_velocity(self, swarm):
        print("speed")
        for i in range(self.swarm_size):
            pbest = copy(swarm[i].attributes['local_best'])
            gbest = self.select_global_best()

            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

            w = round(random.uniform(self.weight_min, self.weight_max), 1)

            for var in range(swarm[i].number_of_variables):
                swarm_var = swarm[i].variables[var]
                self.speed[i][var] = w * self.speed[i][var] \
                                       + (c1 * r1 * (pbest.variables[var] - swarm_var)) \
                                       + (c2 * r2 * (gbest.variables[var] - swarm_var))

class SMPSO_Variant(SMPSO):

    def __init__(self, **kwargs):
        super(SMPSO_Variant, self).__init__(**kwargs)


    def update_velocity(self, swarm):
        for i in range(self.swarm_size):
            pbest = copy(swarm[i].attributes['local_best'])
            gbest = self.select_global_best()

            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

            w = round(random.uniform(self.weight_min, self.weight_max), 1)

            for var in range(swarm[i].number_of_variables):
                swarm_var = swarm[i].variables[var]
                self.speed[i][var] = w * self.speed[i][var] \
                                       + (c1 * r1 * (pbest.variables[var] - swarm_var)) \
                                       + (c2 * r2 * (gbest.variables[var] - swarm_var))


def omopso(problem, population_size, max_evaluations, evaluator):
    return (
        OMOPSO_Variant,
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


def smpso(problem, population_size, max_evaluations, evaluator):
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
