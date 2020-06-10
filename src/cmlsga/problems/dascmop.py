import math

from jmetal.core.problem import FloatProblem

class DASCMOP1(FloatProblem):

    def __init__(self, number_of_variables=30):
        super(DASCMOP1, self).__init__()
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 11

        self.difficulty_factors = [0.5, 0, 0]

        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["x", "y"]

        self.lower_bound = [0.0] * self.number_of_variables
        self.upper_bound = [1.0] * self.number_of_variables


    def evaluate(self, solution):
        num_variables = self.number_of_variables
        x = solution.variables

        sum = 0
        for j in range(2, num_variables + 1):
            yj = x[j - 1] - math.sin(0.5 * math.pi * x[0])
            sum += yj * yj

        f = [
            x[0] + sum,
            1 - math.pow(x[0], 2) + sum
        ]

        solution.objectives[0] = f[0]
        solution.objectives[1] = f[1]

        # Calculate constraint variables
        a = 20
        b = 2 * self.difficulty_factors[0] - 1

        d = 0.5
        if self.difficulty_factors[1] == 0:
            d = 0

        e = 1e+30
        if (self.difficulty_factors[1] != 0):
            e = d - math.log(self.difficulty_factors[1])


        r = 0.5 * self.difficulty_factors[2]

        # Calculate constraints
        constraints = [0] * self.number_of_constraints
        constraints[0] = math.sin(a * math.pi * x[0]) - b
        constraints[1] = (e - sum) * (sum - d)
        constraints[2] = (e - 0) * (0 - d) # sum2 = 0
        if self.difficulty_factors[1] == 1:
            constraints[1] = 1e4 - math.abs(sum - e)

        p_k = [0,1,0,1,2,0,1,2,3]
        q_k = [1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5]
        a_k = 0.3
        b_k = 1.2
        theta_k = -0.25 * math.pi

        for k in range(0, len(p_k)):
            zero = math.pow(((f[0] - p_k[k]) * math.cos(theta_k) - (f[1] - q_k[k]) * math.sin(theta_k)), 2) / a_k
            one = math.pow(((f[1] - p_k[k]) * math.sin(theta_k) + (f[1] - q_k[k]) * math.cos(theta_k)), 2) / b_k
            constraints[2 + k] = zero + one - r

        solution.constraints = constraints

    def get_name(self):
        return "DASCMOP1"
    
