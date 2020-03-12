import operator
import math

from functools import reduce

from jmetal.core.problem import FloatProblem

x = [i for i in range(1, 20)]
m = 1



"""
Shape functions for WFG benchmarks
"""
def linear(x, m):
    result = [x[i - 1] for i in range(1, len(x) - m)]
    result = reduce(operator.mul, result, 1)

    if m != 1:
        result = result * (1 - x[len(x) - m])

    return result


def convex(x, m):
    result = [1 - math.cos(x[i-1] * math.pi * 0.5) for i in range(1, len(x) - m)]
    result = reduce(operator.mul, result, 1)

    if m != 1:
        result = result * math.sin(x[len(x) - m] * math.pi * 0.5)

    return result


def concave(x, m):
    result = [math.sin(x[i-1] * math.pi * 0.5) for i in range(1, len(x) - m)]
    result = reduce(operator.mul, result, 1)

    if m != 1:
        result = result * math.cos(x[len(x) - m] * math.pi * 0.5)

    return result


def mixed(x, A, alpha):
    tmp = math.cos(2 * A * math.pi * x[0] + math.pi * 0.5) / (2 * A * math.pi)

    return math.pow(1 - x[0] - tmp, alpha)


def disc(x, A, alpha, beta):
    tmp = math.cos(A * math.pow(x[0], beta) * math.pi)

    return 1 - math.pow(x[0], alpha) * math.pow(tmp, 2)


"""
Transformations for WFG benchmarks
"""

TRANSFORMATION_EPSILON = 1e-10

def bPoly(y, alpha):
    if not alpha > 0:
        raise Exception("WFG bPoly transformation: alpha must be > 0")

    return correct_to_01(math.pow(y, alpha))


def bFlat(y, A, B, C):
    tmp1 = min(0, math.floor(y - B)) * A * (B - y) / B
    tmp2 = min(0, math.floor(C - y)) * (1 - A) * (y - C) / (1 - C)

    return correct_to_01(A + tmp1 - tmp2)


def sLinear(y, A):
    return correct_to_01(abs(y - A) / abs(math.floor(A - y) + A))


def sDecept(y, A, B, C):
    tmp1 = math.floor(y - A + B) * (1 - C + (A - B) / B) / (A - B)
    tmp2 = math.floor(A + B - y) * (1 - C + (1 - A - B) / B) / (1 - A - B)
    tmp2 = abs(y - A) - B

    return correct_to_01(1 + tmp3 * (tmp1 + tmp2 + 1 / B))


def sMulti(y, A, B, C):
    tmp1 = (4 * A + 2) * math.pi * (0.5 - abs(y - C) / (2 * (math.floor(C - y) + C)))
    tmp2 = 4 * B * math.pow(abs(y - C) / (2 * (math.floor(C - y) + C)), 2)

    return correct_to_01((1 + math.cos(tmp1) + tmp2) / (B + 2))


def rSum(y, w):
    tmp1 = tmp2 = 0
    for i in range(len(y)):
        tmp1 = tmp1 + y[i] * w[i]
        tmp2 = w[i]

    return correct_to_01(tmp1 / tmp2)


def rNonsep(y, A):
    tmp = math.ceil(A / 2)
    denominator = len(y) * tmp * (1 + 2 * A - 2 * tmp) / A

    numerator = 0
    for j in range(len(y)):
        numerator += y[j]
        for k in range(A - 1):
            numerator += abs(y[j] - y[(j + k + 1) % len(y)])

    return correct_to_01(numerator / denominator)


def bParam(y, u, A, B, C):
    tmp = A - (1 - 2 * U) * abs(math.floor(0.5 - u) + A)
    exp = B + (C - B) * v

    return correct_to_01(math.pow(y, exp))


def correct_to_01(a):
    min = 0
    max = 1
    min_epsilon = min - TRANSFORMATION_EPSILON
    max_epsilon = max + TRANSFORMATION_EPSILON

    if (a <= min and a >= min_epsilon) or (a >= min and a <= min_epsilon):
        return min
    elif (a >= max and a <= max_epsilon) or (a <= max and a >= max_epsilon):
        return max
    else:
        return a


"""
WFG problems
"""
class WFG(FloatProblem):

    def __init__(self, k, l, m):
        super(WFG, self).__init__()
        self.k = k
        self.l = l
        self.m = m
        self.d = 1

        self.number_of_variables = k + l
        self.number_of_objectives = m
        self.number_of_constraints = 0

        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = [2 * (i+1) for i in range(self.number_of_variables)]


    # Gets the x vector
    def calculate_x(self, t):
        return [
            max(t[self.m - 1], self.a[i]) * (t[i] - 0.5) + 0.5
            for i in range(0, self.m - 1)
        ] + [t[self.m - 1]]


    def normalise(self, z):
        return [correct_to_01(z[i] / (2 * (i + 1))) for i in range(len(z))]


    def sub_vector(self, z, head, tail):
        """
        Get a subvector of a given vector (Head and tail inclusive)
        """
        return z[int(head):int(tail) + 1].copy()



class WFG1(WFG):

    def __init__(self, k=2, l=4, m=2):
        super(WFG1, self).__init__(k, l, m)

        self.s = [2 * (i + 1) for i in range(self.m)]
        self.a = [i for i in range(self.m - 1)]


    def evaluate(self, solution):
        x = reduce(
            (lambda x,f: f(x)),
            [self.normalise, self.t1, self.t2, self.t3, self.t4, self.calculate_x],
            solution.variables
        )

        result = [0] * self.m
        for m in range(1, self.m):
            result[m] = self.d * x[self.m - 1] + self.s[m - 1] * convex(x, m)
        result[self.m - 1] = self.d * x[self.m - 1] + x[self.m - 1] * mixed(x, 5, 1)

        for i in range(len(result)):
            solution.objectives[i] = result[i]


    def t1(self, z):
        return z[:self.k] + [sLinear(z[i], 0.35) for i in range(self.k, len(z))]

    def t2(self, z):
        return z[:self.k] + [bFlat(z[i], 0.8, 0.75, 0.85) for i in range(self.k, len(z))]

    def t3(self, z):
        return [bPoly(z[i], 0.02) for i in range(0, len(z))]

    def t4(self, z):
        result = [0] * self.m
        w = [2 * (i + 1) for i in range(len(z))]

        for i in range(1, self.m + 1):
            head = (i - 1) * self.k / (self.m - 1) + 1
            tail = i * self.k / (self.m - 1)

            subz = self.sub_vector(z, head-1, tail-1)
            subw = self.sub_vector(w, head-1, tail-1)

            result[i - 1] = rSum(subz, subw)

        head = self.k
        tail = len(z) - 1

        subz = self.sub_vector(z, head, tail)
        subw = self.sub_vector(w, head, tail)

        result[self.m - 1] = rSum(subz, subw)
        return result


    def get_name(self):
        return "WFG1"
