
class Collective(object):

    def __init__(self, algorithm, label):
        self.algorithm = algorithm
        self.label = label

        self.evaluations = 0
        self.solutions = []


    def add_solution(self, solution, max_number=0):
        if max_number != 0:
            if len(self.solutions) < max_number:
                self.solutions.append(solution)
        else:
            self.solutions.append(solution)

 
    def erase(self):
        self.solutions = []
        self.algorithm.solutions = []


    def restart(self):
        self.algorithm.solutions = self.solutions
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
        collective_fitness = 0
        for solution in self.solutions:
            temp_fitness = 0
            for objective in solution.objectives:
                temp_fitness += objective / len(solution.objectives)
            collective_fitness += temp_fitness
        collective_fitness /= len(self.solutions)

        return collective_fitness


    # TODO sort and get by fitness
    def best_solutions(self, num_solutions):
        return self.algorithm.solutions[:num_solutions]


    def init_algorithm(self, *args):
        constructor, kwargs = self.algorithm(*args)

        self.algorithm = constructor(**kwargs)
        self.algorithm.solutions = self.solutions


    def __repr__(self):
        return "Collective {} - {} - {} solutions".format(
            self.label, self.algorithm, len(self.algorithm.solutions))


def avg(l):
    return sum(l) / len(l)
