from jmetal.util.archive import NonDominatedSolutionsArchive, CrowdingDistanceArchive

class Collective(object):

    def __init__(self, algorithm, label):
        self.algorithm = algorithm
        self.label = label

        self.evaluations = 0
        self.solutions = []
        self.archive = NonDominatedSolutionsArchive()


    def add_solution(self, solution, max_number=0):
        if max_number != 0:
            if len(self.solutions) < max_number:
                self.solutions.append(solution)
        else:
            self.solutions.append(solution)

 
    def erase(self):
        self.solutions = []
        self.algorithm.solutions = []
        self.archive = NonDominatedSolutionsArchive()


    def restart(self):
        self.algorithm.solutions = self.solutions
        self.algorithm.archive = CrowdingDistanceArchive(len(self.solutions))
        self.algorithm.population_size = len(self.solutions)

        for s in self.solutions:
            self.algorithm.archive.solution_list.append(s)
            self.algorithm.archive.non_dominated_solution_archive.solution_list.append(s)

        self.algorithm.archive.compute_density_estimator()
        # self.algorithm.init_progress()
        # self.algorithm.evaluations += self.evaluations


    def step(self):
        self.algorithm.step()
        self.evaluations = self.algorithm.evaluations


    def evaluate(self):
        return self.algorithm.evaluate(self.algorithm.solutions)


    def calculate_fitness(self):
        self.solutions = self.algorithm.solutions

        # Average fitness of all solutions (MLS1)
        #collective_fitness = 0
        #for solution in self.solutions:
        #    temp_fitness = 0
        #    for objective in solution.objectives:
        #        temp_fitness += objective / len(solution.objectives)
        #    collective_fitness += temp_fitness
        #collective_fitness /= len(self.solutions)

        for solution in self.solutions:
            self.archive.add(solution)
        collective_fitness = len(self.archive.solution_list)

        return collective_fitness


    def best_solutions(self, num_solutions):
        best_solutions = self.archive.solution_list[:num_solutions]

        # add more solutions if not enough
        num_missing_solutions = num_solutions - len(best_solutions)
        best_solutions.extend(self.solutions[:num_missing_solutions])

        return best_solutions


    def init_algorithm(self, *args):
        constructor, kwargs = self.algorithm(*args)

        self.algorithm = constructor(**kwargs)
        self.algorithm.solutions = self.solutions
        for s in self.solutions:
            self.algorithm.archive.solution_list.append(s)
            self.algorithm.archive.non_dominated_solution_archive.solution_list.append(s)
        self.algorithm.archive.compute_density_estimator()


    def __repr__(self):
        return "Collective {} - {} - {} solutions".format(
            self.label, self.algorithm, len(self.algorithm.solutions))


def avg(l):
    return sum(l) / len(l)
