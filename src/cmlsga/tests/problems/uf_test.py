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

        self.assertEqual(UF1().get_name(), "UF1")
        self.assertAlmostEqual(1.3663694656987078, solution.objectives[0])
        self.assertAlmostEqual(0.4628361455406435, solution.objectives[1])

    def test_uf2(self):
        solution = self.evaluate_solution(UF2, [1] * 30)

        self.assertEqual(UF2().get_name(), "UF2")
        self.assertAlmostEqual(3.1241822036047, solution.objectives[0])
        self.assertAlmostEqual(1.2657634982543, solution.objectives[1])

    def test_uf3(self):
        solution = self.evaluate_solution(UF3, [2] * 30)

        self.assertEqual(UF3().get_name(), "UF3")
        self.assertAlmostEqual(8.531083801519983, solution.objectives[0])
        self.assertAlmostEqual(6.926995381598061, solution.objectives[1])

    def test_uf3_yj_pj_length(self):
        uf3 = UF3()
        solution = self.evaluate_solution(UF3, [1] * 30)
        yj, pj = zip(*[uf3.f(solution.variables, j, uf3.number_of_variables)
                       for j in range(3, uf3.number_of_variables + 1, 2)])

        self.assertEqual(len(yj), len(pj))

    def test_uf4(self):
        solution = self.evaluate_solution(UF4, [1] * 30)

        self.assertEqual(UF4().get_name(), "UF4")
        self.assertAlmostEqual(1.1649783496395125, solution.objectives[0])
        self.assertAlmostEqual(0.1704097832512541, solution.objectives[1])

    def test_uf5(self):

        self.assertEqual(UF5().get_name(), "UF5")



if __name__ == "__main__":
    unittest.main()
