import random
import math
import numpy as np

from jmetal.core.problem import FloatProblem, DynamicProblem
from jmetal.core.solution import FloatSolution

from sympy import sin, pi, cos, floor


class JY(DynamicProblem, FloatProblem):
    def __init__(self):
        super(JY, self).__init__()
        self.number_of_variables = 10
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.tau = 5
        self.nT = 10
        self.time = 0
        self.problem_modified = False

        self.gt = math.sin(0.5 * math.pi * self.time)

        self.upper_bound = [1] + [1 for _ in range(self.number_of_variables - 1)]
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
        x = solution.variables
        f0, f1 = self.eval_objectives(x, self.get_at(), self.get_wt())

        solution.objectives[0] = f0
        solution.objectives[1] = f1

    def eval_objectives(self, x, at, wt):
        sum_gx = sum([math.pow(x[i] - self.gt, 2) for i in range(1, self.number_of_variables)])

        obj1 = (1 + sum_gx) * (x[0] + at * math.sin(wt * math.pi * x[0]))
        obj2 = (1 + sum_gx) * (1 - x[0] + at * math.sin(wt * math.pi * x[0]))

        return obj1, obj2

    def get_differentials(self, s, at, wt):
        sum_gx = sum([(s.xvars[i] - self.gt) ** 2 for i in range(1, self.number_of_variables)])

        f0 = (1 + sum_gx) * (s.xvars[0] + at * sin(wt * pi * s.xvars[0]))
        f1 = (1 + sum_gx) * (1 - s.xvars[0] + at * sin(wt * pi * s.xvars[0]))

        return (f0, f1)


    def get_at(self):
        pass

    def get_wt(self):
        pass

    def get_name(self):
        pass


class JY1(JY):
    def __init__(self):
        super().__init__()

    def get_at(self):
        return 0.05

    def get_wt(self):
        return 6

    def differentials(self, s):
        return self.get_differentials(s, self.get_at(), self.get_wt())

    def pf(self, obj, num_points, time):
        pass

    def get_name(self):
        return "JY1"


class JY2(JY):
    def __init__(self):
        super().__init__()

    def get_at(self):
        return 0.05

    def get_wt(self):
        return math.floor(6 * math.sin(0.5 * math.pi * (self.time - 1)))

    def differentials(self, s):
        return self.get_differentials(s, self.get_at(), self.get_wt())
    
    def pf(self, obj, num_points, time):
        pass

    def get_name(self):
        return "JY2"


class JY3(JY):
    def __init__(self):
        super().__init__()

        self.at = 0.05

    def evaluate(self, solution):
        x = solution.variables
        wt = math.floor(6 * math.sin(0.5 * math.pi * (self.time - 1)))
        alpha = math.floor(100 * math.pow(math.sin(0.5 * math.pi * self.time), 2))
        y0 = abs(x[0] * math.sin((2 * alpha + 0.5) * math.pi * x[0]))

        sum_gx = 0
        for i in range(1, self.number_of_variables):
            yi = x[i]
            yj = 0
            if i == 1:
                yj = y0
            else:
                yj = x[i - 1]

            sum_gx += math.pow(math.pow(yi, 2) - yj, 2)

        solution.objectives[0] = (1 + sum_gx) * (y0 + self.at * math.sin(wt * math.pi * y0))
        solution.objectives[1] = (1 + sum_gx) * (1 - y0 + self.at * math.sin(wt * math.pi * y0))

    def differentials(self, s):
        wt = floor(6 * sin(0.5 * pi * (self.time - 1)))
        alpha = floor(100 * (sin(0.5 * pi * self.time) ** 2))
        y0 = abs(s.xvars[0] * sin((2 * alpha + 0.5) * pi * s.xvars[0]))

        sum_gx = 0
        for i in range(1, self.number_of_variables):
            yi = s.xvars[i]
            yj = 0
            if i == 1:
                yj = y0
            else:
                yj = s.xvars[i - 1]

            sum_gx += ((yi ** 2 ) - yj) ** 2

        f0 = (1 + sum_gx) * (y0 + self.at * sin(wt * pi * y0))
        f1 = (1 + sum_gx) * (1 - y0 + self.at * sin(wt * pi * y0))

        return (f0, f1)


    def pf(self, obj, num_points, time):
        pass


    def get_name(self):
        return "JY3"



class JY4(JY):
    def __init__(self):
        super().__init__()

    def get_at(self):
        return 0.05

    def get_wt(self):
        return math.pow(10, 1 + abs(self.gt))

    def differentials(self, s):
        return self.get_differentials(s, self.get_at(), self.get_wt())

    def pf(self, obj, num_points, time):
        pass

    def get_name(self):
        return "JY4"



class JY5(JY):
    def __init__(self):
        super().__init__()

    def get_at(self):
        return 0.3 * math.sin(0.5 * math.pi * (self.time - 1))

    def get_wt(self):
        return 1

    def differentials(self, s):
        return self.get_differentials(s, self.get_at(), self.get_wt())

    def pf(self, obj, num_points, time):
        pass

    def get_name(self):
        return "JY5"


class JY6(JY):
    def __init__(self):
        super().__init__()
        self.at = 0.1
        self.wt = 3

    def evaluate(self, solution):
        x = solution.variables
        kt = 2 * math.floor(10 * abs(self.gt))
        sum_gx = 0
        for i in range(1, self.number_of_variables):
            yi = x[i] - self.gt
            sum_gx += 4 * math.pow(yi, 2) - math.cos(kt * math.pi * yi) + 1

        solution.objectives[0] = (1 + sum_gx) * (x[0] + self.at * math.sin(self.wt * math.pi * x[0]))
        solution.objectives[1] = (1 + sum_gx) * (1 - x[0] + self.at * math.sin(self.wt * math.pi * x[0]))


    def differentials(self, s):
        kt = 2 * math.floor(10 * abs(self.gt))
        sum_gx = 0
        for i in range(1, self.number_of_variables):
            yi = s.xvars[i] - self.gt
            sum_gx += 4 * (yi ** 2) - cos(kt * pi * yi) + 1

        f0 = (1 + sum_gx) * (s.xvars[0] + self.at * sin(self.wt * pi * s.xvars[0]))
        f1 = (1 + sum_gx) * (1 - s.xvars[0] + self.at * sin(self.wt * pi * s.xvars[0]))

        return (f0, f1)


    def pf(self, obj, num_points, time):
        pass


    def get_name(self):
        return "JY6"


class JY7(JY):
    def __init__(self):
        super().__init__()
        self.at = 0.1
        self.wt = 3

    def evaluate(self, solution):
        x = solution.variables
        alpha_t = 0.2 + 2.8 * abs(self.gt)

        sum_gx = 0
        for i in range(1, self.number_of_variables):
            yi = x[i] - self.gt
            sum_gx += math.pow(yi, 2) - 10 * math.cos(2 * math.pi * yi) + 10

        solution.objectives[0] = (1 + sum_gx) * math.pow(x[0] + self.at * math.sin(self.wt * math.pi * x[0]), alpha_t)
        solution.objectives[1] = (1 + sum_gx) * math.pow(1 - x[0] + self.at * math.sin(self.wt * math.pi * x[0]), alpha_t)


    def differentials(self, s):
        alpha_t = 0.2 + 2.8 * abs(self.gt)
        sum_gx = 0
        for i in range(1, self.number_of_variables):
            yi = s.xvars[i] - self.gt
            sum_gx += (yi ** 2) - 10 * cos(2 * pi * yi) + 10

        f0 = (1 + sum_gx) * ((s.xvars[0] + self.at * sin(self.wt * pi * s.xvars[0])) ** alpha_t)
        f1 = (1 + sum_gx) * ((1 - s.xvars[0] + self.at * sin(self.wt * pi * s.xvars[0])) ** alpha_t)

        return (f0, f1)

        
    def pf(self, obj, num_points, time):
        pass


    def get_name(self):
        return "JY7"



class JY8(JY):
    def __init__(self):
        super().__init__()
        self.at = 0.05
        self.wt = 6

    def evaluate(self, solution):
        x = solution.variables
        beta_t = 10 - 9.8 * abs(self.gt)
        alpha_t = 2 / beta_t

        sum_gx = sum([math.pow(x[i], 2) for i in range(1, self.number_of_variables)])

        solution.objectives[0] = (1 + sum_gx) * math.pow(x[0] + self.at * math.sin(self.wt * math.pi * x[0]), alpha_t)
        solution.objectives[1] = (1 + sum_gx) * math.pow(1 - (x[0] + self.at * math.sin(self.wt * math.pi * x[0])), alpha_t)


    def differentials(self, s):
        beta_t = 10 - 9.8 * abs(self.gt)
        alpha_t = 2 / beta_t

        sum_gx = sum([s.xvars[i] ** 2 for i in range(1, self.number_of_variables)])

        f0 = (1 + sum_gx) * ((s.xvars[0] + self.at * sin(self.wt * pi * s.xvars[0])) ** alpha_t)
        f1 = (1 + sum_gx) * ((1 - (s.xvars[0] + self.at * sin(self.wt * pi * s.xvars[0]))) ** alpha_t)

        return (f0, f1)


    def pf(self, obj, num_points, time):
        pass


    def get_name(self):
        return "JY8"

