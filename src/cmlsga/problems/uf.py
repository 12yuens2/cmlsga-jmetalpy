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

        self.n = n
        self.epsilon = epsilon


    def f(self, x, j, num_variables):
        yj = x[j - 1] - math.sin(6 * math.pi * x[0] + j * math.pi / num_variables)
        hj = 2 * yj * yj - math.cos(4 * math.pi * yj) + 1

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

        hj = (0.5 / self.n + self.epsilon) * abs(math.sin(2 * num_variables * math.pi * x[0]))
        solution.objectives[0] = x[0] + hj + 2 * sum1 / count1
        solution.objectives[1] = 1 - x[0] + hj + 2 * sum2 / count2


    def get_name(self):
        return "UF5"


class UF6(UF1):

    def __init__(self, number_of_variables=30, n=2, epsilon=0.1):
        super(UF6, self).__init__(number_of_variables)

        self.n = n
        self.epsilon = epsilon


    def f(self, x, j, num_variables):
        yj = x[j - 1] - math.sin(6 * math.pi * x[0] + j * math.pi / num_variables)
        pj = math.cos(20 * yj * math.pi / math.sqrt(j))

        return yj, pj

    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        yj1, pj1 = zip(*[self.f(x, j, num_variables) for j in range(3, num_variables + 1, 2)])
        sum1 = sum([yj * yj for yj in yj1])
        prod1 = reduce(operator.mul, pj1, 1)
        count1 = len(yj1)

        yj2, pj2 = zip(*[self.f(x, j, num_variables) for j in range(2, num_variables + 1, 2)])
        sum2 = sum([yj * yj for yj in yj2])
        prod2 = reduce(operator.mul, pj2, 1)
        count2 = len(yj2)

        hj = max(2 * (0.5 / self.n + self.epsilon) * math.sin(2 * num_variables * math.pi * x[0]), 0)

        solution.objectives[0] = x[0] + hj + 2 * (4 * sum1 - 2 * prod1 + 2) / count1
        solution.objectives[1] = 1 - x[0] + hj + 2 * (4 * sum2 - 2 * prod2 + 2) / count2

    def get_name(self):
        return "UF6"


class UF7(UF1):

    def __init__(self, number_of_variables=30):
        super(UF7, self).__init__(number_of_variables)


    def f(self, x, j, num_variables):
        yj = x[j - 1] - math.sin(6 * math.pi * x[0] + j * math.pi / num_variables)

        return yj

    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        yj1 = [self.f(x, j, num_variables) for j in range(3, num_variables + 1, 2)]
        sum1 = sum([yj * yj for yj in yj1])
        count1 = len(yj1)

        yj2 = [self.f(x, j, num_variables) for j in range(2, num_variables + 1, 2)]
        sum2 = sum([yj * yj for yj in yj2])
        count2 = len(yj2)

        yj = math.pow(x[0], 0.2)

        solution.objectives[0] = yj + 2 * sum1 / count1
        solution.objectives[1] = 1 - yj + 2 * sum2 / count2

    def get_name(self):
        return "UF7"


class UF8(UF1):

    def __init__(self, number_of_variables=30):
        super(UF8, self).__init__(number_of_variables)

        self.number_of_objectives = 3

        self.lower_bound = [0.0, 0.0] + [-2.0] * (number_of_variables - 2)
        self.upper_bound = [1.0, 1.0] + [2.0] * (number_of_variables - 2)


    def f(self, x, j, num_variables):
        yj = x[j - 1] - 2 * x[1] * math.sin(2 * math.pi * x[0] + j * math.pi / num_variables)

        return yj

    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        yj1 = [self.f(x, j, num_variables) for j in range(4, num_variables + 1, 3)]
        sum1 = sum([yj * yj for yj in yj1])
        count1 = len(yj1)

        yj2 = [self.f(x, j, num_variables) for j in range(5, num_variables + 1, 3)]
        sum2 = sum([yj * yj for yj in yj2])
        count2 = len(yj2)

        yj3 = [self.f(x, j, num_variables) for j in range(3, num_variables + 1, 3)]
        sum3 = sum([yj * yj for yj in yj3])
        count3 = len(yj3)

        solution.objectives[0] = math.cos(0.5*math.pi*x[0]) * math.cos(0.5*math.pi*x[1]) + 2*sum1/count1
        solution.objectives[1] = math.cos(0.5*math.pi*x[0]) * math.cos(0.5*math.pi*x[1]) + 2*sum2/count2
        solution.objectives[2] = math.sin(0.5*math.pi*x[0]) + 2*sum3/count3


    def get_name(self):
        return "UF8"



class UF9(UF8):

    def __init__(self, number_of_variables=30, epsilon=0.1):
        super(UF9, self).__init__(number_of_variables)

        self.epsilon = epsilon


    def f(self, x, j, num_variables):
        yj = x[j - 1] - 2*x[1]*math.sin(2*math.pi*x[0] + j*math.pi/num_variables)

        return yj

    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        yj1 = [self.f(x, j, num_variables) for j in range(4, num_variables + 1, 3)]
        sum1 = sum([yj * yj for yj in yj1])
        count1 = len(yj1)

        yj2 = [self.f(x, j, num_variables) for j in range(5, num_variables + 1, 3)]
        sum2 = sum([yj * yj for yj in yj2])
        count2 = len(yj2)

        yj3 = [self.f(x, j, num_variables) for j in range(3, num_variables + 1, 3)]
        sum3 = sum([yj * yj for yj in yj3])
        count3 = len(yj3)

        yj = max((1 + self.epsilon) * (1 - 4*(2*x[0] - 1) * (2*x[0] - 1)), 0)

        solution.objectives[0] = 0.5*(yj + 2*x[0]) * x[1] + 2*sum1/count1
        solution.objectives[1] = 0.5*(yj + 2*x[0] + 2) * x[1] + 2*sum2/count2
        solution.objectives[2] = 1 - x[1] + 2*sum3/count3

    def get_name(self):
        return "UF9"

class UF10(UF8):

    def __init__(self, number_of_variables=30):
        super(UF10, self).__init__(number_of_variables)

    def f(self, x, j, num_variables):
        yj = x[j - 1] - 2*x[1]*math.sin(2*math.pi*x[0] + j*math.pi/num_variables)
        hj = 4*yj*yj - math.cos(8*math.pi*yj) + 1

        return hj

    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        yj1 = [self.f(x, j, num_variables) for j in range(4, num_variables + 1, 3)]
        sum1 = sum(yj1)
        count1 = len(yj1)

        yj2 = [self.f(x, j, num_variables) for j in range(5, num_variables + 1, 3)]
        sum2 = sum(yj2)
        count2 = len(yj2)

        yj3 = [self.f(x, j, num_variables) for j in range(3, num_variables + 1, 3)]
        sum3 = sum(yj3)
        count3 = len(yj3)

    def get_name(self):
        return "UF10"
