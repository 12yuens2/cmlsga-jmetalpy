import numpy as np
import math

from jmetal.core.problem import FloatProblem

class MOP1(FloatProblem):

    def __init__(self, number_of_variables=10):
        super(MOP1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = [1.0] * self.number_of_variables


    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - math.sin(0.5 * math.pi * x[0])
            g += -0.9 * t * t + math.pow(abs(t), 0.6)

        return 2 * math.sin(math.pi * x[0]) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - math.sqrt(x[0]))


    def pf(self, obj, num_points):
        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [1 - math.sqrt(i) for i in f1]

        return zip(f1, f2)

    def get_name(self):
        return "MOP1"



class MOP2(MOP1):
    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - math.sin(0.5 * math.pi * x[0])
            g += abs(t) / (1 + math.exp(5 * abs(t)))

        return 10 * math.sin(math.pi * x[0]) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - x[0] * x[0])


    def pf(self, obj, num_points):
        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [1 - i * i for i in f1]

        return zip(f1, f2)

    def get_name(self):
        return "MOP2"


class MOP3(MOP1):
    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - math.sin(0.5 * math.pi * x[0])
            g += abs(t) / (1 + math.exp(5 * abs(t)))

        return 10 * math.sin(0.5 * math.pi * x[0]) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * math.cos(x[0] * math.pi * 0.5)
        solution.objectives[1] = (1 + g) * math.sin(x[0] * math.pi * 0.5)


    def pf(self, obj, num_points):
        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [math.sqrt(1 - i * i) for i in f1]

        return zip(f1, f2)


    def get_name(self):
        return "MOP3"


class MOP4(MOP1):
    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - math.sin(0.5 * math.pi * x[0])
            g += abs(t) / (1 + math.exp(5 * abs(t)))

        return 1 + 10 * math.sin(math.pi * x[0]) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)


        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - math.sqrt(x[0]) * math.pow(math.cos(x[0] * math.pi * 2), 2))


    def pf(self, obj, num_points):
        min_val = 300
        f1 = []
        f2 = []
        for i in range(num_points):
            v1 = 0 + i * 1/(num_points - 1)
            v2 = 1 - math.sqrt(v1) * math.cos(2 * math.pi * v1 * v1)

            if v2 < min_val:
                f1.append(v1)
                f2.append(v2)
                min_val = v2

        return zip(f1, f2)


    def get_name(self):
        return "MOP4"


class MOP5(MOP1):
    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - math.sin(0.5 * math.pi * x[0])
            g += -0.9 * t * t + math.pow(abs(t), 0.6)

        return 2 * abs(math.cos(math.pi * x[0])) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - math.sqrt(x[0]))


    def pf(self, obj, num_points):
        f1 = [0 + i * 1/(num_points - 1) for i in range(num_points)]
        f2 = [1 - math.sqrt(i) for i in f1]

        return zip(f1, f2)


    def get_name(self):
        return "MOP5"


class MOP6(MOP1):

    def __init__(self):
        super(MOP6, self).__init__()
        self.number_of_objectives = 3


    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - x[0] * x[1]
            g += -0.9 * t * t + math.pow(abs(t), 0.6)

        return 2 * math.sin(math.pi * x[0]) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0] * x[1]
        solution.objectives[1] = (1 + g) * x[0] * (1 - x[1])
        solution.objectives[2] = (1 + g) * (1 - x[0])


    def pf(self, obj, num_points):
        points = []
        for i in range(num_points):
            # Generate random points, then normalise
            vec = np.random.uniform(0, 1, 3)
            arr = vec / np.sum(vec)

            points.append(tuple(arr))

        return points


    def get_name(self):
        return "MOP6"


class MOP7(MOP6):
    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - x[0] * x[1]
            g += -0.9 * t * t + math.pow(abs(t), 0.6)

        return 2 * math.sin(math.pi * x[0]) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * math.cos(0.5 * math.pi * x[0]) * math.cos(0.5 * math.pi * x[1])
        solution.objectives[1] = (1 + g) * math.cos(0.5 * math.pi * x[0]) * math.sin(0.5 * math.pi * x[1])
        solution.objectives[2] = (1 + g) * math.sin(0.5 * math.pi * x[0])


    def pf(self, obj, num_points):
        #todo refactor uniform point generation into a mop6 function
        points = []
        for i in range(num_points):
            vec = np.random.uniform(0, 1, 3)
            arr = vec / np.sum(vec)

            points.append(arr)

        norm_points = []
        for arr in points:
            s = math.sqrt(sum([i * i for i in arr]))
            norm = [i / s for i in arr]
            norm_points.append(tuple(norm))

        return norm_points


    def get_name(self):
        return "MOP7"
