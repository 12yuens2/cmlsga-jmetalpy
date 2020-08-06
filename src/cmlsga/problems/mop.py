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


    def get_name(self):
        return "MOP7"
