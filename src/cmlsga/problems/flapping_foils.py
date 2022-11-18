import operator
import math

from functools import reduce

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class FF(FloatProblem):

    def __init__(self, number_of_variables=30):
        super(FF, self).__init__()
        self.number_of_variables = ??
        self.number_of_objectives = ??
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = [-1.0] * self.number_of_variables
        self.upper_bound = [1.0] * self.number_of_variables

    def create_solution():
        if previous solutions exist:
            read them
        else:
            super().create_solution()

    def evaluate(self, solution):
        #solution.variables = x
        # run fortran(x)
        # put fitness function and fortran calls here

        # total value = call other postprocessing python

        #solution.objectives[0] = total value

        #print(solution
        pass

    def get_name(self):
        return "Flapping Foils"

