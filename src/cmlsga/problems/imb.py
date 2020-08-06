import math

from jmetal.core.problem import FloatProblem

class IMB1(FloatProblem):

    def __init__(self, number_of_variables=10):
        super(IMB1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.numer_of_constraints = 0

        self.obj_diretions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = [1.0] * self.number_of_variables


    def eval_g(self, x):
        if 0 <= x[0] and x[0] <= 0.2:
            return 0
        else:
            g = 0.0
            for i in range(1, len(x)):
                t = x[i] - math.sin(0.5 * math.pi * x[0])
                g += 0.5 * (-0.9 * t * t + math.pow(abs(t), 0.6))

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - math.sqrt(x[0]))


    def get_name(self):
        return "IMB1"


class IMB2(IMB1):
    def eval_g(self, x):
        if 0.4 <= x[0] and x[0] <= 0.6:
            return 0
        else:
            g = 0.0
            for i in range(1, len(x)):
                t = x[i] - math.sin(0.5 * math.pi * x[0])
                g += 0.5 * (-0.9 * t * t + math.pow(abs(t), 0.6))

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - x[0])


    def get_name(self):
        return "IMB2"


class IMB3(IMB1):
    def eval_g(self, x):
        if 0.8 <= x[0] and x[0] <= 1:
            return 0
        else:
            g = 0.0
            for i in range(1, len(x)):
                t = x[i] - math.sin(0.5 * math.pi * x[0])
                g += 0.5 * (-0.9 * t * t + math.pow(abs(t), 0.6))

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * math.cos(math.pi * x[0] / 2)
        solution.objectives[1] = (1 + g) * math.sin(math.pi * x[0] / 2)


    def get_name(self):
        return "IMB3"


class IMB4(IMB1):
    def __init__(self):
        super(IMB4, self).__init__()
        self.number_of_objectives = 3
        
    def eval_g(self, x):
        if 2/3 <= x[0] and x[0] <= 1:
            return 0
        else:
            g = 0.0
            for i in range(2, len(x)):
                t = x[i] - (x[0] + x[1]) / 2
                g += -0.9 * t * t + math.pow(abs(t), 0.6)

            return 2 * math.cos(math.pi * x[0] / 2) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0] * x[1]
        solution.objectives[1] = (1 + g) * x[0] * (1 - x[1])
        solution.objectives[2] = (1 + g) * (1 - x[0])


    def get_name(self):
        return "IMB4"


class IMB5(IMB4):
    def eval_g(self, x):
        if 0 <= x[0] and x[0] <= 0.5:
            return 0
        else:
            g = 0.0
            for i in range(2, len(x)):
                t = x[i] - (x[0] + x[1]) / 2
                g += -0.9 * t * t + math.pow(abs(t), 0.6)

            return 2 * math.cos(math.pi * x[0] / 2) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * math.cos(math.pi * x[0] / 2) * math.cos(math.pi * x[1] / 2)
        solution.objectives[1] = (1 + g) * math.cos(math.pi * x[0] / 2) * math.sin(math.pi * x[1] / 2)
        solution.objectives[2] = (1 + g) * math.sin(math.pi * x[0] / 2)


    def get_name(self):
        return "IMB5"


class IMB6(IMB4):
    def eval_g(self, x):
        if 0 <= x[0] and x[0] <= 0.75:
            return 0
        else:
            g = 0.0
            for i in range(2, len(x)):
                t = x[i] - (x[0] + x[1]) / 2
                g += -0.9 * t * t + math.pow(abs(t), 0.6)

            return 2 * math.cos(math.pi * x[0] / 2) * g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0] * x[1]
        solution.objectives[1] = (1 + g) * x[0] * (1 - x[1])
        solution.objectives[2] = (1 + g) * (1 - x[0])


    def get_name(self):
        return "IMB6"


class IMB7(IMB1):
    def eval_g(self, x):
        g = 0.0
        if 0.5 <= x[0] and x[0] <= 0.8:
            for i in range(1, len(x)):
                s = x[i] - math.sin(0.5 * math.pi * x[0])
                g += -0.9 * s * s + math.pow(abs(s), 0.6)

            return g

        else:
            for i in range(1, len(x)):
                t = x[i] - 0.5
                g += math.pow(abs(t), 0.6)

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - math.sqrt(x[0]))


    def get_name(self):
        return "IMB7"



class IMB8(IMB1):
    def eval_g(self, x):
        g = 0.0
        if 0.5 <= x[0] and x[0] <= 0.8:
            for i in range(1, len(x)):
                s = x[i] - math.sin(0.5 * math.pi * x[0])
                g += -0.9 * s * s + math.pow(abs(s), 0.6)

            return g

        else:
            for i in range(1, len(x)):
                t = x[i] - 0.5
                g += math.pow(abs(t), 0.6)

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - x[0])


    def get_name(self):
        return "IMB8"


class IMB9(IMB1):
    def eval_g(self, x):
        g = 0.0
        if 0.5 <= x[0] and x[0] <= 0.8:
            for i in range(1, len(x)):
                s = x[i] - math.sin(0.5 * math.pi * x[0])
                g += -0.9 * s * s + math.pow(abs(s), 0.6)

            return g

        else:
            for i in range(1, len(x)):
                t = x[i] - 0.5
                g += math.pow(abs(t), 0.6)

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * math.cos(math.pi * x[0] / 2)
        solution.objectives[1] = (1 + g) * math.sin(math.pi * x[0] / 2)


    def get_name(self):
        return "IMB9"


class IMB10(IMB4):
    def eval_g(self, x):
        g = 0.0
        if 0.2 <= x[0] and x[0] <= 0.8 and 0.2 <= x[1] and x[1] <= 0.8:
            for i in range(2, len(x)):
                s = x[i] - (x[0] + x[1]) / 2
                g += 2 * (-0.9 * s * s + math.pow(abs(s), 0.6))

            return g

        else:
            for i in range(2, len(x)):
                t = x[i] - (x[0] * x[1])
                g += math.pow(abs(t), 0.6)

            return g


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)

        solution.objectives[0] = (1 + g) * x[0] * x[1]
        solution.objectives[1] = (1 + g) * x[0] * (1 - x[1])
        solution.objectives[2] = (1 + g) * (1 - x[0])


    def get_name(self):
        return "IMB10"


class IMB11(IMB1):
    def __init__(self):
        super(IMB11, self).__init__()
        self.number_of_constraints = 1


    def eval_g(self, x):
        g = 0.0
        for i in range(1, len(x)):
            t = x[i] - x[0]
            g += 0.5 * (-0.9 * t * t + math.pow(abs(t), 0.6))

        return g


    def eval_constraints(self, x, g):
        G = 0.0
        if x[0] > 0.6 and g > 0.001:
            G = 10 * g

        return G


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)
        constraints = self.eval_constraints(x, g)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - x[0])

        solution.constraints[0] = constraints


    def get_name(self):
        return "IMB11"



class IMB12(IMB11):
    def eval_constraints(self, x, g):
        G = 0.0
        if x[0] < 0.2 or x[0] > 0.8 and g > 0.002:
            G = 10 * g

        return G


    def get_name(self):
        return "IMB12"



class IMB13(IMB11):
    def eval_constraints(self, x, g):
        G = 0.0
        if x[0] < 0.2 or x[0] > 0.8 and g > 0.001:
            G = 10 * g

        return G

    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)
        constraints = self.eval_constraints(x, g)

        solution.objectives[0] = (1 + g) * x[0]
        solution.objectives[1] = (1 + g) * (1 - math.pow(x[0], 2))

        solution.constraints[0] = constraints


    def get_name(self):
        return "IMB13"


class IMB14(IMB4):
    def __init__(self):
        super(IMB14, self).__init__()
        self.number_of_constraints = 1


    def eval_g(self, x):
        g = 0
        for i in range(2, len(x)):
            t = x[i] - (x[0] + x[1]) / 2
            g += 0.5 * (-0.9 * t * t + math.pow(abs(t), 0.6))

        return g


    def eval_constraints(self, x, g):
        G = 0.0
        if x[0] > 0.75 and g > 0.001:
            G = 10 * g

        return G


    def evaluate(self, solution):
        x = solution.variables
        g = self.eval_g(x)
        constraints = self.eval_constraints(x, g)

        solution.objectives[0] = (1 + g) * x[0] * x[1]
        solution.objectives[1] = (1 + g) * x[0] * (1 - x[1])
        solution.objectives[2] = (1 + g) * (1 - x[0])

        solution.constraints[0] = constraints


    def get_name(self):
        return "IMB14"
