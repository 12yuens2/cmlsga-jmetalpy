
class Collective(object):

    def __init__(self, algorithm, label):
        self.algorithm = algorithm
        self.label = label

        self.solutions = []


    def add_solution(self, solution):
        self.solutions.append(solution)


    def init_algorithm(self, *args):
        constructor, kwargs = self.algorithm(*args)

        self.algorithm = constructor(**kwargs)
        self.algorithm.solutions = self.solutions


    def __repr__(self):
        return "Collective {} - {} - {} solutions\n".format(
            self.label, self.algorithm, len(self.algorithm.solutions))
