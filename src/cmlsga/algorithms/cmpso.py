import numpy as np
import random

from copy import copy

from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.util.archive import BoundedArchive, CrowdingDistanceArchive, NonDominatedSolutionsArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance


class CMPSO(ParticleSwarmOptimization):

    def __init__(self,
                 problem,
                 swarm_size,
                 termination_criterion=store.default_termination_criteria,
                 swarm_generator = store.default_generator,
                 swarm_evaluator = store.default_evaluator):
        super(CMPSO, self).__init__(problem, swarm_size)
        self.gbests = [None for i in range(0, problem.number_of_objectives)]

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator

        self.dominance_comparator = DominanceComparator()

        self.w_max = 0.9
        self.w_min = 0.5
        self.c1 = self.c2 = self.c3 = 4.0/3.0
        self.change_velocity1 = self.change_velocity2 = -1

        self.archive_size = 100
        self.archive = []

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


    def init_progress(self):
        super().init_progress()
        self.update_archive()


    def initialize_velocity(self, swarms):
        for m in range(self.problem.number_of_objectives):
            for i in range(self.swarm_size):
                for d in range(self.problem.number_of_variables):
                    self.speed[m][i][d] = random.uniform(-self.speed_max[d], self.speed_max[d])


    def initialize_particle_best(self, swarms):
        for swarm in swarms:
            for particle in swarm:
                particle.attributes["local_best"] = copy(particle)


    def initialize_global_best(self, swarms):
        swarms = self.evaluate(swarms)

        for m in range(0, len(swarms)):
            for i in range(0, len(swarms[m])):
                particle = swarms[m][i]
                if (not self.gbests[m]) or (particle.attributes["local_best"].objectives[m] < self.gbests[m].objectives[m]):
                    self.gbests[m] = particle


    def update_archive(self):
        r = []
        for s in self.archive:
            r.append(copy(s))

        # Add elites
        elites = self.elitist_learning_strategy()
        for elite in elites:
            r.append(copy(elite))

        # Add pbest of each particle
        for swarm in self.solutions:
            for particle in swarm:
                r.append(copy(particle.attributes["local_best"]))

        self.archive = self.nondominated_sorting(r)

        # Keep best particles if archive is full
        if len(self.archive) > self.archive_size:
            distance_estimator = CrowdingDistance()
            distance_estimator.compute_density_estimator(self.archive)
            self.archive = sorted(self.archive, reverse=True, key=lambda x: x.attributes["crowding_distance"])[:self.archive_size]


    def nondominated_sorting(self, solutions):
        a = NonDominatedSolutionsArchive()
        for s in solutions:
            a.add(s)

        return a.solution_list


    def select_archive_solution(self):
        return random.choice(self.archive)


    def elitist_learning_strategy(self):
        elites = []
        for solution in self.archive:
            e = solution
            d = random.randint(0, self.problem.number_of_variables - 1)
            Xmax = self.problem.upper_bound[d]
            Xmin = self.problem.lower_bound[d]
            e.variables[d] = e.variables[d] + (Xmax - Xmin) * np.random.normal(0, 1)

            # Limit to upper and lower bounds
            if e.variables[d] > Xmax:
                e.variables[d] = Xmax
            elif e.variables[d] < Xmin:
                e.variables[d] = Xmin

            elites.append(e)

        elites = self.evaluate([elites])

        return elites[0]


    def update_particle_best(self, swarms):
        for m in range(0, len(swarms)):
            for particle in swarms[m]:
                v = particle.objectives[m]
                if v < particle.attributes["local_best"].objectives[m]:
                    particle.attributes["local_best"] = copy(particle)


    def update_global_best(self, swarms):
        for m in range(0, len(swarms)):
            gbest = self.gbests[m]
            for i in range(0, len(swarms[m])):
                particle = swarms[m][i]
                v = particle.objectives[m]

                if v < self.gbests[m].objectives[m]:
                    self.gbests[m] = copy(particle)


    def update_velocity(self, swarms):
        w = self.w_max - (self.w_max - self.w_min) * (self.termination_criterion.evaluations / self.termination_criterion.max_evaluations)

        for m in range(0, len(swarms)):
            gbest = self.gbests[m]

            for i in range(0, len(swarms[m])):
                current_position = swarms[m][i]
                archive_position = self.select_archive_solution()

                pbest = current_position.attributes["local_best"]
                r1 = random.random()
                r2 = random.random()
                r3 = random.random()

                for d in range(0, self.problem.number_of_variables):
                    current = current_position.variables[d]
                    v1 = (self.c1 * r1 * (pbest.variables[d] - current))
                    v2 = (self.c2 * r2 * (gbest.variables[d] - current))
                    v3 = (self.c3 * r3 * (archive_position.variables[d] - current))

                    self.speed[m][i][d] = w * self.speed[m][i][d] + v1 + v2 + v3



    def update_position(self, swarms):
        for m in range(len(swarms)):
            for i in range(len(swarms[m])):
                particle = swarms[m][i]

                for d in range(particle.number_of_variables):
                    particle.variables[d] += self.speed[m][i][d]

                    # Limit speed to max velocities
                    if particle.variables[d] < self.problem.lower_bound[d]:
                        particle.variables[d] = self.problem.lower_bound[d]
                        self.speed[m][i][d] *= self.change_velocity1

                    if particle.variables[d] > self.problem.upper_bound[d]:
                        particle.variables[d] = self.problem.upper_bound[d]
                        self.speed[m][i][d] *= self.change_velocity2


    def step(self):
        super().step()
        self.update_archive()


    def perturbation(self, swarms):
        return swarms


    def get_observable_data(self):
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "ALL_SOLUTIONS": [s for swarm in self.solutions for s in swarm]        
        }

    def get_result(self):
        return self.archive


    def get_name(self):
        return "CMPSO"


