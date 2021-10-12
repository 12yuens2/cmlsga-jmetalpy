import math

from jmetal.problem.multiobjective.fda import *

class FDA1e(FDA1):

    def __init__(self):
        super(FDA1e, self).__init__()

    def pf(self, obj, num_points, time):
        f1 = [i * 1/(num_points - 1) for i in range(0, num_points)]
        f2 = [1 - math.sqrt(i) for i in f1]

        return zip(f1, f2)


class FDA2e(FDA2):

    def __init__(self):
        super(FDA2e, self).__init__()

    def pf(self, obj, num_points, time):
        ht = 0.75 + 0.7 * math.sin(0.5 * math.pi * time)

        f1 = []
        f2 = []
        for i in range(0, num_points):
            mod = 0
            if ht <= 1:
                mod = ht
            else:
                mod = ht + 15 * math.pow(ht - 1, 2)

            i = i * 1/(num_points - 1)
            f1.append(i)
            f2.append(1 - math.pow(i, mod))

        return zip(f1, f2)


class FDA3e(FDA3):

    def __init__(self):
        super(FDA3e, self).__init__()

    def pf(self, obj, num_points, time):
        gt = abs(math.sin(0.5 * math.pi * time))
        ht = math.pow(10, 2 * math.sin(0.5 * math.pi * time))

        f1 = [i * 1/(num_points - 1) for i in range(0, num_points)]
        f2 = [(1 - math.sqrt(i / (1 + gt))) * (1 + gt) for i in f1]

        return zip(f1, f2)


class FDA4e(FDA4):

    def __init__(self):
        super(FDA4e, self).__init__()

    def pf(self, obj, num_points, time):
        gt = abs(math.sin(0.5 * math.pi * time))

        f1 = [i * 1/(num_points - 1) + gt for i in range(0, num_points)]
        f2 = [1 - i + 2 * gt for i in f1]

        return zip(f1, f2)


class FDA5e(FDA5):

    def __init__(self):
        super(FDA5e, self).__init__()

    def pf(self, obj, num_points, time):
        gt = abs(math.sin(0.5 * math.pi * time))

        f1 = [i * 1/(num_points - 1) + gt for i in range(0, num_points)]
        f2 = [1 - i + 2 * gt for i in f1]

        return zip(f1, f2)


