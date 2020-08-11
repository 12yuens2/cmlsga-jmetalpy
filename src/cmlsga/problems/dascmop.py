"""
Adapted from pymoo https://github.com/msu-coinlab/pymoo/blob/master/pymoo/problems/multi/dascmop.py

"""

import math
import numpy as np

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

DIFFICULTIES = [
    (0.25, 0., 0.), (0., 0.25, 0.), (0., 0., 0.25), (0.25, 0.25, 0.25),
    (0.5, 0., 0.), (0., 0.5, 0.), (0., 0., 0.5), (0.5, 0.5, 0.5),
    (0.75, 0., 0.), (0., 0.75, 0.), (0., 0., 0.75), (0.75, 0.75, 0.75),
    (0., 1.0, 0.), (0.5, 1.0, 0.), (0., 1.0, 0.5), (0.5, 1.0, 0.5)
]


class DASCMOP(FloatProblem):
    def __init__(self, number_of_objectives, number_of_constraints, difficulty):
        super(DASCMOP, self).__init__()
        self.number_of_variables = 30
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x" "y"]

        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = [1.0] * self.number_of_variables

        self.eta, self.zeta, self.gamma = self.get_difficulty_values(difficulty)

    def get_difficulty_values(self, difficulty):
        if isinstance(difficulty, int):
            self.difficulty = difficulty
            if not (1 <= difficulty <= len(DIFFICULTIES)):
                raise Exception(
                    "Difficulty must be 1 <= difficulty <= {}, but is {}".format(
                    len(DIFFICULTIES), difficulty)
                )
            return DIFFICULTIES[difficulty-1]
        else:
            self.difficulty = -1
            return difficulty

    def g1(self, x):
        g = 0
        for j in range(2, self.number_of_variables + 1):
            yj = x[j - 1] - np.sin(0.5 * np.pi * x[0])
            g += yj * yj
        return g


    def g2(self, x):
        g = 0
        for j in range(self.number_of_objectives + 1, self.number_of_variables + 1):
            z = x[j - 1] - 0.5
            yj = z ** 2 - np.cos(20 * np.pi * z)
            g += yj

        return (self.number_of_variables - self.number_of_objectives + 1) + g


    def g3(self, x):
        g = 0
        for j in range(4, self.number_of_variables + 1):
            yj = x[j - 1] - cos(0.25 * (j - 1) / self.number_of_variables * np.pi * (x[0] + x[2]))
            g += yj * yj

        return g


    def evaluate(self, solution):
        pass

    def get_name(self):
        return "DASCMOP"


class DASCMOP1(DASCMOP):
    def __init__(self, difficulty):
        super().__init__(2, 11, difficulty)

    def evaluate_constraints(self, x, f0, f1, g):
        a = 20
        b = 2 * self.eta - 1

        d = 0.5
        if self.zeta == 0:
            d = 0

        e = 1e+30
        if self.zeta != 0:
            e = d - np.log(self.zeta)

        r = 0.5 * self.gamma

        # Calculate constraints
        constraints = [0] * self.number_of_constraints
        constraints[0] = np.sin(a * np.pi * x[0]) - b
        constraints[1] = (e - g) * (g - d)
        constraints[2] = (e - 0) * (0 - d) # sum2 = 0
        if self.zeta == 1:
            constraints[1] = 1e4 - np.abs(g - e)

        p_k = [0,1,0,1,2,0,1,2,3]
        q_k = [1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5]
        a_k2 = 0.3
        b_k2 = 1.2
        theta_k = -0.25 * math.pi

        for k in range(0, len(p_k)):
            zero = math.pow(((f0 - p_k[k]) * math.cos(theta_k) - (f1 - q_k[k]) * math.sin(theta_k)), 2) / a_k2
            one = math.pow(((f1 - p_k[k]) * math.sin(theta_k) + (f1 - q_k[k]) * math.cos(theta_k)), 2) / b_k2
            constraints[2 + k] = zero + one - r

        return constraints


    def set_solution(self, solution, f0, f1, constraints):
        """
        Set the objective and constraints of the solution
        """
        solution.objectives[0] = f0
        solution.objectives[1] = f1

        solution.constraints = constraints


    def evaluate(self, solution): #X, out, *args, **kwargs):
        x = solution.variables
        g = self.g1(x)

        f0 = x[0] + g
        f1 = 1.0 - x[0] ** 2 + g
        constraints = self.evaluate_constraints(x, f0, f1, g)

        self.set_solution(solution, f0, f1, constraints)


    def get_name(self):
        return "DASCMOP1({})".format(self.difficulty)


class DASCMOP2(DASCMOP1):
    def evaluate(self, solution):
        x = solution.variables
        g = self.g1(x)

        f0 = x[0] + g
        f1 = 1.0 - np.sqrt(x[0]) + g
        constraints = self.evaluate_constraints(x, f0, f1, g)

        self.set_solution(solution, f0, f1, constraints)


    def get_name(self):
        return "DASCMOP2({})".format(self.difficulty)


class DASCMOP3(DASCMOP1):
    def evaluate(self, solution):
        x = solution.variables
        g = self.g1(x)

        f0 = x[0] + g
        f1 = 1.0 - np.sqrt(x[0]) + 0.5 * np.abs(np.sin(5 * np.pi * x[0])) + g
        constraints = self.evaluate_constraints(x, f0, f1, g)

        self.set_solution(solution, f0, f1, constraints)


    def get_name(self):
        return "DASCMOP3({})".format(self.difficulty)


class DASCMOP4(DASCMOP1):
    def evaluate(self, solution):
        x = solution.variables
        g = self.g2(x)

        f0 = x[0] + g
        f1 = 1.0 - x[0] ** 2 + g
        constraints = self.evaluate_constraints(x, f0, f1, g)

        self.set_solution(solution, f0, f1, constraints)


    def get_name(self):
        return "DASCMOP4({})".format(self.difficulty)


class DASCMOP5(DASCMOP1):
    def evaluate(self, solution):
        x = solution.variables
        g = self.g2(x)

        f0 = x[0] + g
        f1 = 1.0 - np.sqrt(x[0]) + g
        constraints = self.evaluate_constraints(x, f0, f1, g)

        self.set_solution(solution, f0, f1, constraints)


    def get_name(self):
        return "DASCMOP5({})".format(self.difficulty)


class DASCMOP6(DASCMOP1):
    def evaluate(self, solution):
        x = solution.variables
        g = self.g2(x)

        f0 = x[0] + g
        f1 = 1.0 - np.sqrt(x[0]) + 0.5 * np.abs(np.sin(5 * np.pi * x[0])) + g
        constraints = self.evaluate_constraints(x, f0, f1, g)

        self.set_solution(solution, f0, f1, constraints)

    def get_name(self):
        return "DASCMOP6({})".format(self.difficulty)


#class DASCMOP7(DASCMOP):
#    def __init__(self, difficulty_factors):
#        super().__init__(3, 7, difficulty_factors)
#
#    def constraints(self, X, f0, f1, f2, g):
#        a = 20.
#        b = 2. * self.eta - 1.
#        d = 0.5 if self.zeta != 0 else 0
#        if self.zeta > 0:
#            e = d - np.log(self.zeta)
#        else:
#            e = 1e30
#        r = 0.5 * self.gamma
#
#        x_k = np.array([[1.0, 0., 0., 1.0 / np.sqrt(3.0)]])
#        y_k = np.array([[0., 1.0, 0., 1.0 / np.sqrt(3.0)]])
#        z_k = np.array([[0., 0., 1.0, 1.0 / np.sqrt(3.0)]])
#
#        c = np.zeros((X.shape[0], 3 + x_k.shape[1]))
#
#        c[:, 0] = np.sin(a * np.pi * X[:, 0]) - b
#        c[:, 1] = np.cos(a * np.pi * X[:, 1]) - b
#        if self.zeta == 1:
#            c[:, 2:3] = 1e-4 - np.abs(e - g)
#        else:
#            c[:, 2:3] = (e - g) * (g - d)
#
#        c[:, 3:] = (f0 - x_k) ** 2 + (f1 - y_k) ** 2 + (f2 - z_k) ** 2 - r ** 2
#        return -1 * c
#
#    def _evaluate(self, X, out, *args, **kwargs):
#        g = self.g2(X)
#
#        f0 = X[:, 0:1] * X[:, 1:2] + g
#        f1 = X[:, 1:2] * (1.0 - X[:, 0:1]) + g
#        f2 = 1 - X[:, 1:2] + g
#
#        out["F"] = np.column_stack([f0, f1, f2])
#        out["G"] = self.constraints(X, f0, f1, f2, g)
#
#
#class DASCMOP8(DASCMOP7):
#
#    def objectives(self, X, g):
#        f0 = np.cos(0.5 * np.pi * X[:, 0:1]) * np.cos(0.5 * np.pi * X[:, 1:2]) + g
#        f1 = np.cos(0.5 * np.pi * X[:, 0:1]) * np.sin(0.5 * np.pi * X[:, 1:2]) + g
#        f2 = np.sin(0.5 * np.pi * X[:, 0:1]) + g
#        return np.column_stack([f0, f1, f2])
#
#    def _evaluate(self, X, out, *args, **kwargs):
#        g = self.g2(X)
#        F = self.objectives(X, g)
#        out["F"] = F
#        out["G"] = self.constraints(X, F[:, 0:1], F[:, 1:2], F[:, 2:3], g)
#
#
#class DASCMOP9(DASCMOP8):
#    def _evaluate(self, X, out, *args, **kwargs):
#        g = self.g3(X)
#        F = self.objectives(X, g)
#        out["F"] = F
#        out["G"] = self.constraints(X, F[:, 0:1], F[:, 1:2], F[:, 2:3], g)
