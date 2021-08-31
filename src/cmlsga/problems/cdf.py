
import math
import numpy as np

from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution

class CDF(DynamicProblem, FloatProblem):
    def __init__(self, number_of_constraints):
        super(CDF, self).__init__()
        self.number_of_variables = 30
        self.number_of_objectives = 2
        self.number_of_constraints = number_of_constraints

        self.tau = 5
        self.nT = 10
        self.time = 1
        self.problem_modified = False

        self.gt = math.sin(0.5 * math.pi * self.time)

        self.upper_bound = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]

    def update(self, *args, **kwargs):
       counter = kwargs["COUNTER"]
       self.time = (1.0 / self.nT) * math.floor(counter * 1.0 / self.tau)
       self.gt = math.sin(0.5 * math.pi * self.time)

       self.problem_modified = True

    def the_problem_has_changed(self):
        return self.problem_modified

    def clear_changed(self):
        self.problem_modified = False

    def evaluate(self, solution):
        pass


class CDF1(CDF):
    def __init__(self):
        super().__init__(2)


    def evaluate(self, solution):
        x = solution.variables
        constraints = self.eval_constraints(x)

        size_j1 = sum_j1 = size_j2 = sum_j2 = 0

        for i in range(1, self.number_of_variables):
            y = x[i] - math.pow(x[0], (0.5 * (1 + 3 * (i - 1) / (self.number_of_variables - 2)) + abs(self.gt)))
            if i % 2 == 1:
                size_j1 += 1
                sum_j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                sum_j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * sum_j1
        solution.objectives[1] = math.pow(1 - x[0], 2) + (2 / size_j2) * sum_j2

        solution.constraints = constraints


    def eval_constraints(self, x):
        constraints = [0, 0]
        k1 = 0.5 * (1 - x[0]) - math.pow(1 - x[0], 2)
        k2 = 0.25 * math.sqrt(1 - x[0]) - 0.5 * (1 - x[0])

        constraints[0] = x[1] - math.pow(x[0], 0.5 * (2 + abs(self.gt)) - np.sign(k1) * math.sqrt(abs(k1)))
        constraints[1] = x[3] - math.pow(x[0], 0.5 * (2 + 3 * (4 - 2) / (self.number_of_variables - 2)) +
                                         abs(self.gt) - np.sign(k2) * math.sqrt(abs(k2)))

        return constraints


    def pf(self, obj, num_points, time):
        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [math.pow(1 - i, 2) for i in f1]

        return zip(f1, f2)


    def get_name(self):
        return "CDF1"


class CDF2(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        constraints = self.eval_constraints(x)

        size_j1 = sum_j1 = size_j2 = sum_j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
            h = 0

            if i == 2:
                if (y < 3 / 2 * (1 - sqrt(2) / 2)):
                    h = abs(y)
                else:
                    h = 0.125 + math.pow(y - 1, 2)
            else:
                h = pow(y - self.gt, 2)

            if i % 2 == 1:
                size_j1 += 1
                sum_j1 += h
            else:
                size_j2 += 1
                size_j2 += h

        solution.objectives[0] = x[0] + sum_j1
        solution.objectives[1] = 1 - x[0] + sum_j2

        solution.constraints = constraints


    def eval_constraints(self, x):
        k = x[1] - math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - 0.5 * x[0] + 0.25

        return k / (1 + math.pow(math.e, 4 * abs(k)))

    def pf(self, obj, num_points, time):
        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = []
        for i in f1:
            v = 0
            if i <= 0.5: v = 1 - i
            elif i <= 0.75: v = -0.5 * i + 0.75
            else: v = 1 - i + 0.125

            f2.append(v)

        return zip(f1, f2)
            

class CDF3(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1 for _ in range(self.number_of_variables)]
        self.lower_bounds = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        constraints = self.eval_constraints(x)

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.power(x[0], (0.5 * (2 + 3 * (i - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        maxf = (0.5 / self.nT + 0.1) * abs(math.sin(2 * self.nT * math.pi * x[0]))

        solution.objectives[0] = x[0] + (2 / size_j1) * j1 + maxf
        solution.objectives[1] = 1 - x[0] + (2 / size_j2) * j2 + maxf

        solution.constraints = constraints


    def eval_constraints(self, x):
        k = x[1] - math.pow(x[0], (0.5 * (2 + 3 * (2 - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

        return k


    def pf(self, obj, num_points, time):
        f1 = f2 = []
        for i in range(self.nT * 2):
            j = i / (2 * self.nT)
            f1.append(j)
            f2.append(1 - j)

        return zip(f1, f2)


class CDF4(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.power(x[0], (0.5 * (1 + 3 * (i - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * j1
        solution.objectives[1] = 1 - math.pow(x[0], 2) + (2 / size_j2) * j2

        solution.constraints = self.eval_constraints(solution.objectives)


    def eval_constraints(self, x):
        k = x[0] + x[1] - 1 * abs(math.sin(self.nT * math.pi * (x[0] - x[1] + 1))) - 1

        return k


    def pf(self, obj, num_points, time):
        f1 = f2 = []
        for i in range(num_points):
            v1 = 0 + i * 1/(num_points - 1)
            v2 = 1 - math.pow(i, 2)

            if i + v2 - abs(math.sin(self.nT * math.pi * (i - v2 + 1))) - 1 > -0.000001:
                f1.append(v1)
                f2.append(v2)

        return zip(f1, f2)



class CDF5(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        j1 = j2 = h = 0

        for i in range(2, self.number_of_variables + 1):
            if i % 2 == 1:
                y = x[i - 1] - 0.8 * x[0] * math.cos(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables)) - self.gt
            else:
                y = x[i - 1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables)) - self.gt

            if i == 2:
                if y < (3 / 2 * (1 - sqrt(2) / 2)):
                    h = abs(y)
                else:
                    h = 0.125 + math.pow(y - 1, 2)
            else:
                h = 2 * math.pow(y, 2) - math.cos(4 * math.pi * y) + 1
 
            if i % 2 == 1:
                j1 += h
            else:
                j2 += h

        solution.objectives[0] = x[0] + j1 + abs(self.gt)
        solution.objectives[1] = 1 - x[0] + j2 + abs(self.gt)

        solution.constraints = self.eval_constraints(x)


    def eval_constraints(self, x):
        k = x[1] + 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - 0.5 * x[0] + 0.25 - self.gt

        return k


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time)
        f1 = f2 = []

        for i in range(num_points):
            v1 = 0 + i * 1/(num_points - 1)
            v2 = 0
            if v1 <= 0.5:
                v2 = 1 - i
            elif v1 <= 0.75:
                v2 = -0.5 * i + 0.75
            else:
                v2 = 1 - i + 0.125

            f1.append(v1 + abs(gt))
            f2.append(v2 + abs(gt))

        return zip(f1, f2)



class CDF6(CDF):
    def __init__(self):
        super().__init__(2)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        j1 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.power(x[0], (0.5 * (1 + 3 * (i - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

            if i % 2 == 1:
                y = x[i - 1] - 0.8 * x[0] * math.cos(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables)) - abs(gt)
                j1 += math.pow(y, 2)
            else:
                y = x[i - 1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
                if i != 2 and i != 4:
                    y -= abs(gt)
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + j1 + abs(gt)
        solution.objectives[1] = math.pow(1 - x[0], 2) + j2 + abs(gt)

        solution.constraints = self.eval_constraints(x)


    def eval_constraints(self, x):
        cons = [0, 0]
        k1 = 0.5 * (1 - x[0]) - math.pow(1 - x[0], 2)
        k2 = 0.25 * math.sqrt(1 - x[0]) - 0.5 * (1 - x[0])

        cons[0] = x[1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - np.sign(k1) * math.sqrt(abs(k1))
        cons[1] = x[3] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (4 * math.pi / self.number_of_variables)) - np.sign(k2) * math.sqrt(abs(k2))

        return cons


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time)
        f1 = f2 = []

        for i in range(num_points):
            v1 = 0 + i * 1/(num_points - 1)
            v2 = 0
            if v1 <= 0.5:
                v2 = math.pow(1 - i, 2)
            elif v1 <= 0.75:
                v2 = -0.5 * (1 - i)
            else:
                v2 = 0.25 * math.sqrt(1 - i)

            f1.append(v1 + abs(gt))
            f2.append(v2 + abs(gt))

        return zip(f1, f2)


class CDF7(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - self.gt - math.power(x[0], (0.5 * (1 + 3 * (i - 2) / (self.number_of_variables - 2))))
            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + abs(self.gt) + (2 / size_j1) * j1
        solution.objectives[1] = 1 - x[0] + abs(self.gt) + (2 / size_j2) * j2

        solution.constraints = self.eval_constraints(solution.objectives)


    def eval_constraints(self, x):
        k = x[0] + x[1] - 2 * abs(gt) - 1 * abs(math.sin(self.nT * math.pi * (x[0] - x[1] + 1))) - 1

        return k


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time)
        f1 = f2 = []
        for i in range(abs(gt), 1 + abs(gt), 1 / num_points):
            v = 1 - i + 2 * abs(gt)

            if i + v - 2 * abs(gt) - abs(math.sin(self.nT * math.pi * (i - v + 1))) > -0.000001:
                f1.append(i)
                f2.append(v)

        return zip(f1, f2)
