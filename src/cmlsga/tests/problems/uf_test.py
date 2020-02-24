import unittest

from cmlsga.problems.uf import *


class TestUFFunctions(unittest.TestCase):

    def create_solution(self, lower_bound, upper_bound, num_objectives, num_constraints, values):
        solution = FloatSolution(lower_bound, upper_bound,
                                          num_objectives, num_constraints)
        solution.variables = values

        return solution


    def evaluate_solution(self, problem, variable_values):
        p = problem()
        solution = self.create_solution(p.lower_bound, p.upper_bound,
                                        p.number_of_objectives, p.number_of_constraints,
                                        variable_values)
        p.evaluate(solution)
        return solution

    
    def test_uf1(self):
        solution = self.evaluate_solution(UF1, [1] * 30)

        self.assertAlmostEqual(1.3663694656987078, solution.objectives[0])
        self.assertAlmostEqual(0.4628361455406435, solution.objectives[1])

    def test_uf2(self):
        solution = self.evaluate_solution(UF2, [1] * 30)

        self.assertAlmostEqual(3.1241822036047, solution.objectives[0])
        self.assertAlmostEqual(1.2657634982543, solution.objectives[1])


if __name__ == "__main__":
    unittest.main()
