import random
import math
import numpy as np

from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution

class CDF(DynamicProblem, FloatProblem):
    def __init__(self, number_of_constraints):
        super(CDF, self).__init__()
        self.number_of_variables = 10
        self.number_of_objectives = 2
        self.number_of_constraints = number_of_constraints

        self.tau = 5
        self.nT = 10
        self.time = 0
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
                if (y < 3 / 2 * (1 - math.sqrt(2) / 2)):
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

        return [k / (1 + math.pow(math.e, 4 * abs(k)))]

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

    def get_name(self):
        return "CDF2"
            

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
            y = x[i - 1] - math.pow(x[0], (0.5 * (2 + 3 * (i - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

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

        return [k]


    def pf(self, obj, num_points, time):
        f1 = []
        f2 = []
        for i in range(self.nT * 2):
            j = i / (2 * self.nT)
            f1.append(j)
            f2.append(1 - j)

        return zip(f1, f2)


    def get_name(self):
        return "CDF3"


class CDF4(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.pow(x[0], (0.5 * (1 + 3 * (i - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

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

        return [k]


    def pf(self, obj, num_points, time):
        f1 = []
        f2 = []
        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1.0 / (num_points - 1)):
            v2 = 1 - math.pow(i, 2)

            if i + v2 - abs(math.sin(self.nT * math.pi * (i - v2 + 1))) - 1 > -0.000001:
                f1.append(i)
                f2.append(v2)

        return zip(f1, f2)


    def get_name(self):
        return "CDF4"



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
                if y < (3 / 2 * (1 - math.sqrt(2) / 2)):
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

        return [k]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v2 = 0
            if i <= 0.5:
                v2 = 1 - i
            elif i <= 0.75:
                v2 = -0.5 * i + 0.75
            else:
                v2 = 1 - i + 0.125

            f1.append(i + abs(gt))
            f2.append(v2 + abs(gt))

        return zip(f1, f2)


    def get_name(self):
        return "CDF5"



class CDF6(CDF):
    def __init__(self):
        super().__init__(2)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        j1 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.pow(x[0], (0.5 * (1 + 3 * (i - 2) / (self.number_of_variables - 2)) + abs(self.gt)))

            if i % 2 == 1:
                y = x[i - 1] - 0.8 * x[0] * math.cos(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables)) - abs(self.gt)
                j1 += math.pow(y, 2)
            else:
                y = x[i - 1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
                if i != 2 and i != 4:
                    y -= abs(self.gt)
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + j1 + abs(self.gt)
        solution.objectives[1] = math.pow(1 - x[0], 2) + j2 + abs(self.gt)

        solution.constraints = self.eval_constraints(x)


    def eval_constraints(self, x):
        cons = [0, 0]
        k1 = 0.5 * (1 - x[0]) - math.pow(1 - x[0], 2)
        k2 = 0.25 * math.sqrt(1 - x[0]) - 0.5 * (1 - x[0])

        cons[0] = x[1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - np.sign(k1) * math.sqrt(abs(k1))
        cons[1] = x[3] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (4 * math.pi / self.number_of_variables)) - np.sign(k2) * math.sqrt(abs(k2))

        return cons


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v2 = 0
            if i <= 0.5:
                v2 = math.pow(1 - i, 2)
            elif i <= 0.75:
                v2 = 0.5 * (1 - i)
            else:
                v2 = 0.25 * math.sqrt(1 - i)

            f1.append(i + abs(gt))
            f2.append(v2 + abs(gt))

        return zip(f1, f2)


    def get_name(self):
        return "CDF6"


class CDF7(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - self.gt - math.pow(x[0], (0.5 * (1 + 3 * (i - 2) / (self.number_of_variables - 2))))
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
        k = x[0] + x[1] - 2 * abs(self.gt) - 1 * abs(math.sin(self.nT * math.pi * (x[0] - x[1] + 1))) - 1

        return [k]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        f1 = []
        f2 = []
        for i in range(0, 2 * self.nT + 1):
            v1 = i / (2 * self.nT) + abs(gt)
            v2 = (1 - i / (2 * self.nT)) + abs(gt)

            f1.append(v1)
            f2.append(v2)

        return zip(f1, f2)


    def get_name(self):
        return "CDF7"



class CDF8(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.pow(x[0], (0.5 * (2 + 3 * (i - 2) / (self.number_of_variables - 2))))
            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * j1
        solution.objectives[1] = 1 - (0.5 + abs(self.gt)) * math.pow(x[0], 0.5 + abs(self.gt)) + (2 / size_j2) * j2

        solution.constraints = self.eval_constraints(solution.objectives)


    def eval_constraints(self, x):
        k = x[1] + math.pow(x[0], 0.5) - 1 * math.sin(2 * math.pi * math.pow(x[0], 0.5) - x[1] + 1) - 1

        return [k]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        mt = 0.5 + abs(gt)
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = 1 - (mt * math.pow(i, mt))

            if v + math.sqrt(i) - math.sin(2 * math.pi * (math.pow(i, 0.5) - v + 1)) - 1 > -0.000001:
                f1.append(i)
                f2.append(v)

        return zip(f1, f2)

    def get_name(self):
        return "CDF8"


class CDF9(CDF):
    def __init__(self):
        super().__init__(2)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables

        j1 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = 0
            if i % 2 == 1:
                y = x[i - 1] - 0.8 * x[0] * math.cos(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
                j1 += math.pow(y, 2)
            else:
                y = x[i - 1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + j1 + abs(self.gt)
        solution.objectives[1] = math.pow(1 - math.pow(0.5 + abs(self.gt) * x[0], 0.5 + abs(self.gt)), 2) + j2 + abs(self.gt)

        solution.constraints = self.eval_constraints(x)


    def eval_constraints(self, x):
        mt = 0.5 + abs(self.gt)

        v = 0.5 * math.pow(1 - math.pow(mt * x[0], mt), 1) - math.pow(1 - math.pow(mt * x[0], mt), 2)
        v2 = 0.25 * math.pow(1 - min(1, math.pow(mt * x[0], mt)), 0.5) - 0.5 * math.pow(1 - math.pow(mt * x[0], mt), 1)

        return [
            x[1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - np.sign(v) * math.sqrt(abs(v)),
            x[3] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (4 * math.pi / self.number_of_variables)) - np.sign(v2) * math.sqrt(abs(v2))
        ]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        mt = 0.5 + abs(gt)
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = 0
            if 1 - math.pow(mt * i, mt) > 0:
                if math.pow(mt * i, mt) <= 0.5:
                    v = math.pow(1 - math.pow(mt * i, mt), 2)
                elif math.pow(mt * i, mt) <= 0.75:
                    v = 0.5 * math.pow(1 - math.pow(mt * i, mt), 1)
                else:
                    v = 0.25 * math.pow(1 - math.pow(mt * i, mt), 0.5)

                f1.append(i + abs(gt))
                f2.append(v + abs(gt))

        return zip(f1, f2)

    def get_name(self):
        return "CDF9"




class CDF10(CDF):
    def __init__(self):
        super().__init__(2)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        j1 = j2 = 0
        mt = 0.5 + abs(self.gt)

        for i in range(2, self.number_of_variables + 1):
            y = h = 0
            if i % 2 == 1:
                y = x[i - 1] - math.cos(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
            else:
                y = x[i - 1] - math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))

            if i == 2 or i == 4:
                h = math.pow(y, 2)
            else:
                h = 2 * math.pow(y, 2) - math.cos(4 * math.pi * y) + 1

            if i % 2 == 1:
                j1 += h
            else:
                j2 += h


        solution.objectives[0] = x[0] + j1
        solution.objectives[1] = math.pow(1 - x[0], mt) + j2

        solution.constraints = self.eval_constraints(x)


    def eval_constraints(self, x):
        v = 0.5 * (1 - x[0]) - math.pow(1 - x[0], 2)
        v2 = 0.25 * math.sqrt(1 - x[0]) - 0.5 * (1 - x[0])

        return [
            x[1] - math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - np.sign(v) * math.sqrt(abs(v)),
            x[3] - math.sin(6 * math.pi * x[0] + (4 * math.pi / self.number_of_variables)) - np.sign(v2) * math.sqrt(abs(v2))
        ]

    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        mt = 0.5 + abs(gt)
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = 0
            if i <= 0.5:
                v = math.pow(1 - i, mt)
            elif i < 0.75:
                v = math.pow(1 - i, mt) - math.pow(1 - i, 2) + 0.5 * (1 - i)
            elif i <= 1:
                v = math.pow(1 - i, mt) - math.pow(1 - i, 2) + 0.25 * math.sqrt(1 - i)

            f1.append(i)
            f2.append(v)

        return zip(f1, f2)

    def get_name(self):
        return "CDF10"



class CDF11(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        j1 = j2 = 0
        mt = 0.5 + abs(self.gt)
        maxf = (0.5 / self.nT + 0.1) * abs(math.sin((2 * self.nT * x[0] + self.gt) * math.pi))

        for i in range(2, self.number_of_variables + 1):
            y = h = 0
            if i % 2 == 1:
                y = x[i - 1] - 0.8 * x[0] * math.cos(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
            else:
                y = x[i - 1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))

            if i == 2:
                if y < (3 / 2 * (1 - math.sqrt(2) / 2)):
                    h = abs(y)
                else:
                    h = 0.125 + math.pow(y - 1, 2)
            else:
                h = math.pow(y, 2) - math.cos(4 * math.pi * y) + 1

            if i % 2 == 1:
                j1 += h
            else:
                j2 += h


        solution.objectives[0] = x[0] + j1 + maxf
        solution.objectives[1] = 1 - x[0] + j2 + maxf

        solution.constraints = self.eval_constraints(x)


    def eval_constraints(self, x):
        k = x[1] - 0.8 * x[0] * math.sin(6 * math.pi * x[0] + (2 * math.pi / self.number_of_variables)) - 0.5 * x[0] + 0.25

        return [k]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        f1 = []
        f2 = []

        for i in range(0, 2 * self.nT + 1):
            v = (i - gt) / (2 * self.nT)
            v2 = 0
            if v > 0:
                if   v <= 0.5:  v2 = 1 - v
                elif v <= 0.75: v2 = -0.5 * v + 0.75
                elif v <= 1:    v2 = 1 - v + 0.125

                f1.append(v)
                f2.append(v2)

        return zip(f1, f2)

    def get_name(self):
        return "CDF11"


class CDF12(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0
        mt = 0.5 + abs(self.gt)

        for i in range(2, self.number_of_variables + 1):
            y = 0
            if i % 2 == 1:
                y = x[i - 1] - math.sin((6 * math.pi * x[0]) + (i * math.pi / self.number_of_variables))
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                y = x[i - 1] - math.cos((6 * math.pi * x[0]) + (i * math.pi / self.number_of_variables))
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * j1
        solution.objectives[1] = 1 - math.pow(x[0], mt) + (2 / size_j2) * j2

        solution.constraints = self.eval_constraints(solution.objectives)


    def eval_constraints(self, x):
        N = 2
        k = x[1] + math.sqrt(x[0]) - 1 * math.sin(N * math.pi * (math.sqrt(x[0]) - x[1] + 1)) - 1

        return [k / (1 + math.exp(4 * abs(k)))]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        mt = 0.5 + abs(gt)
        N = 2
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = (1 - math.pow(i, mt))
            k = v + math.sqrt(i) - math.sin(N * math.pi * (math.sqrt(i) - v + 1)) - 1

            if k / (1 + math.exp(4 * abs(k))) > -0.000001:
                f1.append(i)
                f2.append(v)

        return zip(f1, f2)

    def get_name(self):
        return "CDF12"


class CDF13(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]

        self.time_vector = 5 * [0]


    def update(self, *args, **kwargs):
        counter = kwargs["COUNTER"]
        self.time = (1.0 / self.nT) * math.floor(counter * 1.0 / self.tau)
        self.gt = math.sin(0.5 * math.pi * self.time)

        i = random.randint(0, 4)
        self.time_vector[i] += 0.1

        self.problem_modified = True


    def evaluate(self, solution):
        x = solution.variables
        gt = [math.sin(0.5 * math.pi * t) for t in self.time_vector]
        kt = math.ceil(self.number_of_variables * gt[0])
        ht4 = 0.5 + abs(gt[3])
        ht5 = 0.5 + abs(gt[4])

        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin(6 * math.pi * x[0] + (i + kt) * math.pi / self.number_of_variables) - gt[1]
            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + abs(gt[2]) + (2 / size_j1) * j1
        solution.objectives[1] = 1 - ht4 * math.pow(x[0], ht5) + abs(gt[2]) + (2 / size_j2) * j2

        solution.constraints = self.eval_constraints(solution.objectives)


    def eval_constraints(self, x):
        gt = [math.sin(0.5 * math.pi * t) for t in self.time_vector]
        ht4 = 0.5 + abs(gt[3])
        ht5 = 0.5 + abs(gt[4])

        k = x[1] + ht4 * math.pow(x[0], ht5) - math.sin(self.nT * math.pi * (ht4 * math.pow(x[0], ht5) - x[1] + 1)) - 1

        return [k]


    def pf(self, obj, num_points, time):
        gt = [math.sin(0.5 * math.pi * t) for t in self.time_vector]
        ht4 = 0.5 + abs(gt[3])
        ht5 = 0.5 + abs(gt[4])
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = 1 - ht4 * math.pow(i, ht5) + abs(gt[2])
            h = ht4 * math.pow(i + abs(gt[2]), ht5)
            k = v + h - math.sin(2 * math.pi * (h - v + 1)) - 1
            if k > -0.00001:
                f1.append(i + abs(gt[2]))
                f2.append(v)

        return zip(f1, f2)

    def get_name(self):
        return "CDF13"



class CDF14(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [0 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.pow(x[0], (0.5 * (1 + (3 * (i - 2) / (self.number_of_variables - 2)))))
            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * j1
        solution.objectives[1] = 1 - x[0] + (2 / size_j2) * j2

        solution.constraints = self.eval_constraints(solution.objectives)


    def eval_constraints(self, x):
        k = x[0] + x[1] - abs(math.sin(self.nT * math.pi * (x[0] - x[1] + 1))) - 1 + abs(self.gt)

        return [k]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = 1 - i
            k = i + v - abs(math.sin(self.nT * math.pi * (i - v + 1))) - 1 + abs(gt)
            if k > -0.000001:
                f1.append(i)
                f2.append(v)

        return zip(f1, f2)


    def get_name(self):
        return "CDF14"



class CDF15(CDF):
    def __init__(self):
        super().__init__(1)

        self.upper_bounds = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bounds = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin(6 * math.pi * x[0] + (i * math.pi / self.number_of_variables))
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
        k = x[1] + math.pow(x[0], 2) - 1 * math.sin(self.nT * math.pi * (math.pow(x[0], 2) - x[1] + 1 + self.gt)) - 1

        return [k]


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time/10)
        ht = 0.5 + abs(gt)
        N = 2
        f1 = []
        f2 = []

        for i in np.arange(0, 1 + (1 / (num_points - 1)), 1 / (num_points - 1)):
            v = 1 - math.pow(i, 2)
            k = math.pow(i, 2) + v - math.sin(N * math.pi * (math.pow(i, 2) - v + 1 + gt)) - 1
            if k > -0.000001:
                f1.append(i)
                f2.append(v)

        return zip(f1, f2)

    def get_name(self):
        return "CDF15"
