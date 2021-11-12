import math
import random

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.operator.crossover import SBXCrossover, DifferentialEvolutionCrossover
from jmetal.util.archive import BoundedArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.density_estimator import CrowdingDistance


class HEIA(GeneticAlgorithm):

    def __init__(self,
                 problem,
                 population_size,
                 mutation,
                 crossover,
                 selection,
                 termination_criterion,
                 population_generator = store.default_generator,
                 population_evaluator = store.default_evaluator,
                 NA = 20):

        super(HEIA, self).__init__(
            problem,
            population_size,
            population_size,
            mutation,
            crossover,
            selection,
            termination_criterion,
            population_generator,
            population_evaluator
        )

        self.NA = 20
        self.theta = 0.9
        self.archive = BoundedArchive(100, DominanceComparator, CrowdingDistance())


    def create_initial_solutions(self):
        solutions = [self.population_generator.new(self.problem) for _ in range(self.population_size)]
        solutions = self.evaluate(solutions)
        for s in solutions:
            self.archive.add(s)

        self.archive.compute_density_estimator()

        return self.archive.solution_list


    def step(self):
        sorted_solutions = sorted(self.archive.solution_list, reverse=True, key=lambda s: s.attributes["crowding_distance"])
        best_solutions = sorted_solutions[:self.NA]

        sum_distance = sum([
            1 if math.isinf(s.attributes["crowding_distance"])
            else s.attributes["crowding_distance"]
            for s in self.solutions
        ])

        p1 = []
        p2 = []
        for bs in best_solutions:
            for _ in range(self.num_clones(bs, sum_distance)):
                clone = bs.__copy__()
                if random.random() < 0.5:
                    p1.append(clone)
                else:
                    p2.append(clone)

        p1 = self.evolve_sbx(p1, best_solutions)
        p2 = self.evolve_de(p2, best_solutions)

        for s in p1 + p2:
            s = self.mutation_operator.execute(s)
            self.archive.add(s)

        return self.archive.solution_list
        
        #mating_population = self.selection(self.solutions)
        #offspring_population = self.reproduction(mating_population)
        #offspring_population = self.evaluate(offspring_population)

        #self.solutions = self.replacement(self.solutions, offspring_population)


    def num_clones(self, solution, sum_distance):
        d = solution.attributes["crowding_distance"]
        crowding_distance = 1 if math.isinf(d) else d

        num = math.ceil(self.population_size * crowding_distance / sum_distance)
        return num


    def evolve_sbx(self, clones, best_solutions):
        crossover_operator = SBXCrossover(1)
        offspring = []
        for c in clones:
            other_parent = best_solutions[random.randint(0, len(best_solutions) - 1)]
            offspring.extend(crossover_operator.execute([c, other_parent]))

        return offspring

    def evolve_de(self, clones, best_solutions):
        crossover_operator = DifferentialEvolutionCrossover(1, 0.5, 0.5)
        offspring = []
        for c in clones:
            crossover_operator.current_individual = c
            mating = [c]
            if random.random() < self.theta:
                mating.extend(self.closest_neighbours(c, clones))
            else:
                mating.extend(random.sample(best_solutions, 2))

        return crossover_operator.execute(mating)


    def closest_neighbours(self, c, clones):
        clones.remove(c)

        objective = random.randint(0, self.problem.number_of_objectives - 1)
        neighbours = sorted([(abs(c.objectives[objective] - n.objectives[objective]), n) for n in clones])[:2]

        return [neighbours[0][1], neighbours[1][1]]


    def get_result(self):
        return self.archive.solution_list


    def get_name(self):
        return "HEIA"
