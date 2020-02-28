import operator
import math

from functools import reduce

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class UF1(FloatProblem):

    def __init__(self, number_of_variables=30):
        super(UF1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = [0.0] + [-1.0] * (self.number_of_variables - 1)
        self.upper_bound = [1.0] * self.number_of_variables


    def f(self, x, j, num_variables):
        return math.pow(x[j - 1] - math.sin(6.0 * math.pi * x[0] + j * math.pi / num_variables), 2)


    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        y1 = [self.f(x, j, num_variables) for j in range(3, num_variables + 1, 2)]
        sum1 = sum(y1)
        count1 = len(y1)

        y2 = [self.f(x, j, num_variables) for j in range(2, num_variables + 1, 2)]
        sum2 = sum(y2)
        count2 = len(y2)

        solution.objectives[0] = x[0] + 2.0 * sum1 / count1
        solution.objectives[1] = 1.0 - math.sqrt(x[0]) + 2.0 * sum2 / count2


    def get_name(self):
        return "UF1"


class UF2(UF1):

    def __init__(self, number_of_variables=30):
        super(UF2, self).__init__(number_of_variables)


    def f(self, g, x, j, num_variables):
        return x[j-1] - (0.3 * x[0] * x[0] * math.cos(24 * math.pi * x[0] + 4 * j * math.pi / num_variables) + 0.6 * x[0]) * g(6 * math.pi * x[0] + j * math.pi / num_variables)


    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        y1 = [self.f(math.cos, x, j, num_variables) for j in range(3, num_variables + 1, 2)]
        sum1 = sum(y1)
        count1 = len(y1)

        y2 = [self.f(math.sin, x, j, num_variables) for j in range(2, num_variables + 1, 2)]
        sum2 = sum(y2)
        count2 = len(y2)

        solution.objectives[0] = x[0] + 2.0 * sum1 / count1
        solution.objectives[1] = 1 - math.sqrt(x[0]) + 2.0 * sum2 / count2


    def get_name(self):
        return "UF2"


class UF3(UF1):

    def __init__(self, number_of_variables=30):
        super(UF3, self).__init__(number_of_variables)

        self.lower_bound = [0] * number_of_variables


    def f(self, x, j, num_variables):
        yj = x[j - 1] - math.pow(x[0], 0.5 * (1 + 3.0 * (j - 2) / (num_variables - 2)))
        pj = math.cos(20 * yj * math.pi / math.sqrt(j))

        return yj * yj, pj


    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        yj1, pj1 = zip(*[self.f(x, j, num_variables) for j in range(3, num_variables + 1, 2)])
        sum1 = sum(yj1)
        prod1 = reduce(operator.mul, pj1, 1)
        count1 = len(yj1)

        yj2, pj2 = zip(*[self.f(x, j, num_variables) for j in range(2, num_variables + 1, 2)])
        sum2 = sum(yj2)
        prod2 = reduce(operator.mul, pj2, 1)
        count2 = len(yj2)

        solution.objectives[0] = x[0] + 2 * (4 * sum1 - 2 * prod1 + 2) / count1
        solution.objectives[1] = 1 - math.sqrt(x[0]) + 2 * (4 * sum2 - 2 * prod2 + 2) / count2


    def get_name(self):
        return "UF3"


class UF4(UF1):

    def __init__(self, number_of_variables=30):
        super(UF4, self).__init__(number_of_variables)

        self.lower_bound = [0.0] + [-2.0] * (number_of_variables - 1)
        self.upper_bound = [1.0] + [2.0] * (number_of_variables - 1)


    def f(self, x, j, num_variables):
        yj = x[j - 1] - math.sin(6 * math.pi * x[0] + j * math.pi / num_variables)
        hj = abs(yj) / (1 + math.exp(2 * abs(yj)))

        return hj


    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        hj1 = [self.f(x, j, num_variables) for j in range(3, num_variables + 1, 2)]
        sum1 = sum(hj1)
        count1 = len(hj1)

        hj2 = [self.f(x, j, num_variables) for j in range(2, num_variables + 1, 2)]
        sum2 = sum(hj2)
        count2 = len(hj2)

        solution.objectives[0] = x[0] + 2 * sum1 / count1
        solution.objectives[1] = 1 - x[0] * x[0] + 2 * sum2 / count2


    def get_name(self):
        return "UF4"
    

class UF5(UF1):

    def __init__(self, number_of_variables=30, n=10, epsilon=0.1):
        super(UF5, self).__init__(number_of_variables)

        self.n = 10
        self.epsilon = 0.1


    def get_name(self):
        return "UF5"
