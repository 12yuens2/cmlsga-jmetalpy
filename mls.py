import kmeans
import math
import time

from collective import Collective
from copy import copy

from jmetal.core.algorithm import Algorithm
from jmetal.config import store
from jmetal.problem.multiobjective.zdt import ZDT1
from jmetal.util.archive import BoundedArchive
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.solutions.comparator import DominanceComparator
from jmetal.util.solutions.evaluator import SparkEvaluator


from typing import TypeVar

S = TypeVar('S')
R = TypeVar('R')

class MultiLevelSelection(Algorithm[S, R]):

    def __init__(self,
                 problem=ZDT1(),
                 number_of_collectives=6,
                 num_new_collectives=2,
                 population_size=600,
                 algorithms=[],
                 mls_mode=7,
                 max_evaluations=30000,
                 population_generator = store.default_generator,
                 population_evaluator = store.default_evaluator):
        super(MultiLevelSelection, self).__init__()

        self.problem = problem
        self.number_of_collectives = number_of_collectives
        self.num_new_collectives = num_new_collectives
        self.population_size = population_size

        self.algorithms = algorithms
        self.coevolution_amount = len(algorithms)
        self.mls_mode = mls_mode

        self.max_evaluations = max_evaluations
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.generations = 0

        self.pareto_front = BoundedArchive(1000, DominanceComparator(), CrowdingDistance())
        self.collectives = self.generate_collectives()


    def generate_collectives(self):
        collectives = self.initialise_temp_collectives()
        print(collectives)
        return collectives


    def initialise_temp_collectives(self):
        subpopulation_sizes = self.get_subpopulation_sizes()

        collectives = []
        for i in range(0, self.coevolution_amount):
            algorithm = self.algorithms[i]
            collective_size = subpopulation_sizes[i]

            population = [
                self.population_generator.new(self.problem)
                for _ in range(collective_size)
            ]

            for collective in self.assign_collectives(algorithm, population):
                collectives.append(collective)

        return collectives


    def assign_collectives(self, algorithm, population):
        labels = self.cluster_population(population)

        collectives = []
        for unique_label in set(labels):

            collective = Collective(algorithm, unique_label)
            for label,solution in zip(labels, population):
                if label == unique_label:
                    collective.add_solution(solution)

            collective.init_algorithm(self.problem,
                                      len(collective.solutions),
                                      self.max_evaluations,
                                      self.population_evaluator)

            collectives.append(collective)

        return collectives


    def cluster_population(self, population):
        labels = []
        if self.number_of_collectives > 1:
            labels = kmeans.Clustering(population, self.number_of_collectives)
        elif self.number_of_collectives == 1:
            labels = [1 for _ in range(0, self.population_size)]
        else:
            raise SystemError("Error: number of collectives < 1")

        return labels


    def get_subpopulation_sizes(self):
        subpopulation_sizes = [
            int(self.population_size / self.coevolution_amount)
            for _ in range(0, self.coevolution_amount)
        ]

        # Spread remainder over the subpopulations
        for i in range(0, self.population_size % self.coevolution_amount):
            subpopulation_sizes[i] += 1

        return subpopulation_sizes


    def create_initial_solutions(self):
        self.collectives = self.generate_collectives()

        solutions = []
        for collective in self.collectives:
            solutions.extend(collective.algorithm.solutions)

        return solutions


    def evaluate(self, solution_list):
        # TODO refactor, convert to list if given a single solution
        if not isinstance(solution_list, list):
            solution_list = [solution_list]

        for collective in self.collectives:
            collective.evaluate()

        for solution in solution_list:
            add = self.pareto_front.add(copy(solution))

        return self.pareto_front.solution_list


    def init_progress(self):
        for collective in self.collectives:
            collective.algorithm.init_progress()


    def stopping_condition_is_met(self):
        self.generations += 1
        return self.generations >= 100

        #evaluations = sum([
        #    collective.evaluations for collective in self.collectives
        #])

        #print("Evaluations: {}\n".format(evaluations))
        #return  evaluations > self.max_evaluations


    def step(self):
        self._update_collectives()
        for i in range(self.num_new_collectives):
            self._replace_worst_collective()


    def _update_collectives(self):
        start = time.time()
        for collective in self.collectives:
            collective.step()
            print("Collective: {}, solutions: {}, evaluations: {}".format(
                collective.algorithm.get_name(),
                len(collective.algorithm.solutions),
                collective.algorithm.evaluations))

            self.solutions = self.evaluate(collective.algorithm.solutions)
        print("Pareto front: {}".format(len(self.pareto_front.solution_list)))
        print("Time taken: {}".format(time.time() - start))


    def _replace_worst_collective(self):
        worst_collective = self._get_worst_collective()
        worst_collective_size = len(worst_collective.solutions)

        worst_collective.erase()
        self.collectives.remove(worst_collective)

        num_solutions = math.ceil(worst_collective_size / len(self.collectives))
        for collective in self.collectives:
            for best_solution in collective.best_solutions(num_solutions):
                worst_collective.add_solution(best_solution, worst_collective_size)

        worst_collective.restart()
        self.collectives.append(worst_collective)


    def _get_worst_collective(self):
        worst_collective = None
        fitness = 1e10

        for collective in self.collectives:
            collective_fitness = collective.calculate_fitness()
            #print("Collective: {}, fitness: {}".format(collective, collective_fitness))
            if collective_fitness < fitness:
                worst_collective = collective
                fitness = collective_fitness

        print("Replace {}, fitness: {}".format(worst_collective, fitness))
        return worst_collective


    def update_progress(self):
        for collective in self.collectives:
            collective.algorithm.update_progress()

    def get_observable_data(self):
        pass

    def get_result(self):
        return self.pareto_front.solution_list

    def get_name(self):
        pass
