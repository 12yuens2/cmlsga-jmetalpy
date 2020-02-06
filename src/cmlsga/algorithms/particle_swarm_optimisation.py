from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation, UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.termination_criterion import StoppingByEvaluations

class OMOPSO_Variant(OMOPSO):

    def __init__(self, problem, swarm_size):
        super(OMOPSO_Variant, self).__init__(problem, swarm_size)


    def update_velocity(self, swarm):
        for i in range(self.swarm_size):
            pbest = copy(swarm[i].attributes['local_best'])
            gbest = self.select_global_best()

            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

            w = round(random.uniform(self.weight_min, self.weight_max), 1)

            for var in range(swarm[i].number_of_variables):
                swarm_var = swarm[i].variables[var]
                self.speed[i][var] = w * self.speed[i][var] \
                                       + (c1 * r1 * (pbest.variables[var] - swarm_var)) \
                                       + (c2 * r2 * (gbest.variables[var] - swarm_var))



class SMPSO_Variant(SMPSO):

    def __init__(self, problem, swarm_size):
        super(SMPSO_Variant, self).__init__(problem, swarm_size)


    def update_velocity(self, swarm):
        for i in range(self.swarm_size):
            pbest = copy(swarm[i].attributes['local_best'])
            gbest = self.select_global_best()

            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)

            w = round(random.uniform(self.weight_min, self.weight_max), 1)

            for var in range(swarm[i].number_of_variables):
                swarm_var = swarm[i].variables[var]
                self.speed[i][var] = w * self.speed[i][var] \
                                       + (c1 * r1 * (pbest.variables[var] - swarm_var)) \
                                       + (c2 * r2 * (gbest.variables[var] - swarm_var))


def omopso(problem, population_size, max_evaluations, evaluator):
    return (
        OMOPSO_Variant,
        {
            "problem": problem,
            "swarm_size": population_size,
            "epsilon": 0.0075,
            "uniform_mutation": UniformMutation(
                probability=1.0 / problem.number_of_variables,
                perturbation=0.5
            ),
            "non_uniform_mutation": NonUniformMutation(
                probability=1.0 / problem.number_of_variables,
                perturbation=0.5,
                max_iterations=int(max_evaluations / population_size)
            ),
            "leaders": CrowdingDistanceArchive(100),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations),
            "swarm_evaluator": evaluator
        }
    )


def smpso(problem, population_size, max_evaluations, evaluator):
    return (
        SMPSO_Variant,
        {
            "problem": problem,
            "swarm_size": population_size,
            "mutation": PolynomialMutation(
                probability=1.0 / problem.number_of_variables,
                distribution_index=20
            ),
            "leaders": CrowdingDistanceArchive(100),
            "termination_criterion": StoppingByEvaluations(max=max_evaluations)
        }
    )
