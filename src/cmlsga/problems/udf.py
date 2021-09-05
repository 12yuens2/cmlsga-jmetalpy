import random
import math
import numpy as np

from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution

class UDF(DynamicProblem, FloatProblem):
    def __init__(self):
        super(UDF, self).__init__()
        self.number_of_variables = 30
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.tau = 5
        self.nT = 2
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


class UDF1(UDF):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin((6 * math.pi * x[0]) + (i * math.pi / self.number_of_variables)) - self.gt

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + abs(self.gt) + (2 / size_j1) * j1
        solution.objectives[1] = 1 - x[0] + abs(self.gt) + (2 / size_j2) * j2


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time)

        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [1 - i + 2 * abs(gt) for i in f1]

        return zip(f1, f2)


    def get_name(self):
        return "UDF1"



class UDF2(UDF1):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - self.gt - math.pow(x[0], (0.5 * (2 + 3 * (i - 2) / (self.number_of_variables - 2)) + self.gt))

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + abs(self.gt) + (2 / size_j1) * j1
        solution.objectives[1] = 1 - x[0] + abs(self.gt) + (2 / size_j2) * j2


    def get_name(self):
        return "UDF2"


class UDF3(UDF):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = mj1 = size_j2 = j2 = mj2 = 0
        maxf = (1 / self.nT + 2 * 0.1) * (math.sin(2 * self.nT * math.pi * x[0]) - abs(2 * self.nT * self.gt))
        if maxf < 0:
            maxf = 0

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin((6 * math.pi * x[0]) + (i * math.pi / self.number_of_variables))

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
                mj1 *= math.cos((20 * y * math.pi) / math.sqrt(i))
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)
                mj2 *= math.cos((20 * y * math.pi) / math.sqrt(i))

        solution.objectives[0] = x[0] + (2 / size_j1) * math.pow(4 * j1 - 2 * mj1 + 2, 2) + maxf
        solution.objectives[1] = 1 - x[0] + (2 / size_j2) * math.pow(4 * j2 - 2 * mj2 + 2, 2) + maxf


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time)
        f1 = f2 = []

        for i in range(0, 1 + 1 / (num_points - 1), 1 / (num_points - 1)):
            if math.sin(2 * self.nT * math.pi * i) < abs(2 * self.nT * gt):
                f1.append(i)
                f2.append(1 - i)

        return zip(f1, f2)


    def get_name(self):
        return "UDF3"


class UDF4(UDF):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0
        mt = 0.5 + abs(self.gt)
        kt = ceil(self.number_of_variables * self.gt)

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin((6 * math.pi * x[0]) + ((i + kt) * math.pi) / self.number_of_variables)

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * j1
        solution.objectives[1] = 1 - (mt * math.pow(x[0], mt)) + (2 / size_j2) * j2


    def pf(self, obj, num_points, time):
        gt = math.sin(0.5 * math.pi * time)
        mt = 0.5 + abs(gt)

        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [1 - mt * math.pow(i, mt)]

        return zip(f1, f2)


    def get_name(self):
        return "UDF4"



class UDF5(UDF4):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0
        mt = 0.5 + abs(self.gt)

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - self.gt - math.pow(x[0], (0.5 * (2 + 3 * (i - 2) / (self.number_of_variables - 2)) + self.gt))

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + (2 / size_j1) * j1
        solution.objectives[1] = 1 - (mt * math.pow(x[0], mt)) + (2 / size_j2) * j2


    def get_name(self):
        return "UDF5"



class UDF6(UDF):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0
        h = (1 / (2 * self.nT) + 0.1) * abs(math.sin(2 * self.nT * math.pi * x[0]) - abs(2 * self.nT * self.gt))
        mt = 0.5 + abs(self.gt)

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.sin(6 * math.pi * x[0] + i * math.pi / self.number_of_variables)

            if i % 2 == 1:
                size_j1 += 1
                j1 += (2 * math.pow(y, 2)) - math.cos(4 * math.pi * y) + 1
            else:
                size_j2 += 1
                j2 += (2 * math.pow(y, 2)) - math.cos(4 * math.pi * y) + 1

        solution.objectives[0] = x[0] + h (2 / size_j1) * j1
        solution.objectives[1] = 1 - mt * x[0] + h + (2 / size_j2) * j2


    def pf(self, obj, num_points, time):
        f1 = f2 = []
        gt = math.sin(0.5 * math.pi * time)
        mt = 0.5 + abs(gt)
        pf_res = 6

        for i in range(0, 2* self.nT + 1):
            if abs(gt) <= math.pow(10, -pf_res):
                t1 = i / 2 * self.nT
                t2 = 1 - (i / 2 * self.nT) * 0.5

                f1.append(t1)
                f2.append(t2)

            elif i % 2 == 0:
                t1 = i / 2 * self.nT + ((1 / 2 * self.nT) + 0.1) * (abs(2 * self.nT * gt) - 1) + (1 / 4 * self.nT)
                t2 = 1 - (i / (2 * self.nT) + (1 / (4 * self.T))) * mt + ((1 / 2 * self.nT) + 0.1) * (abs(2 * self.nT * gt) - 1)

                f1.append(t1)
                f2.append(t2)

        return zip(f1, f2)


    def get_name(self):
        return "UDF6"



class UDF8(UDF):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [1 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-1 for _ in range(self.number_of_variables - 1)]

        self.time_vector = 5 * [0]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0
        gt = [math.sin(0.5 * math.pi * i) for i in self.time_vector]
        kt = ceil(self.number_of_variables * gt[0])
        ht4 = 0.5 + abs(gt[3])
        ht5 = 0.5 + abs(gt[4])

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

    def pf(self, obj, num_points, time):
        gt3 = math.sin(0.5 * math.pi * self.time_vector[2])
        gt4 = math.sin(0.5 * math.pi * self.time_vector[3])
        gt5 = math.sin(0.5 * math.pi * self.time_vector[4])
        ht4 = 0.5 + abs(gt4)
        ht5 = 0.5 + abs(gt5)

        f1 = f2 = []
        for i in range(0 + abs(gt3), 1 + abs(gt3) + (1 / (num_points - 1)), 1 / (num_points - 1)):
            f1.append(i)
            f2.append(1 - ht4 * math.pow(i - abs(gt3), ht5) + abs(gt3))

        return zip(f1, f2)


    def get_name(self):
        return "UDF8"



class UDF9(UDF8):
    def __init__(self):
        super().__init__()

        self.upper_bound = [1] + [2 for _ in range(self.number_of_variables - 1)]
        self.lower_bound = [0] + [-2 for _ in range(self.number_of_variables - 1)]


    def evaluate(self, solution):
        x = solution.variables
        size_j1 = j1 = size_j2 = j2 = 0
        gt = [math.sin(0.5 * math.pi * i) for i in self.time_vector]
        kt = ceil(self.number_of_variables * gt[0])
        ht4 = 0.5 + abs(gt[3])
        ht5 = 0.5 + abs(gt[4])

        for i in range(2, self.number_of_variables + 1):
            y = x[i - 1] - math.pow(x[0], (0.5 * (2 + 3 * (i - 2) / (self.number_of_variables - 2) + gt[0]))) - gt[1]

            if i % 2 == 1:
                size_j1 += 1
                j1 += math.pow(y, 2)
            else:
                size_j2 += 1
                j2 += math.pow(y, 2)

        solution.objectives[0] = x[0] + abs(gt[2]) + (2 / size_j1) * j1
        solution.objectives[1] = 1 - ht4 * math.pow(x[0], ht5) + abs(gt[2]) + (2 / size_j2) * j2


    def get_name(self):
        return "UDF9"


