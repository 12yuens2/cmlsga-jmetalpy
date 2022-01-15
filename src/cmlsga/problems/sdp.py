


from jemtal.core.problem import FloatProblem, DynamicProblem


class SDP(DynamicProblem, FloatProblem):

    def __init__(self):
        super(SDP, self).__init__()

        self.tau = 5
        self.nT = 10
        self.time = 0
        self.problem_modified = False

        self.upper_bound = [1 for _ in range(self.number_of_variables)]
        self.lower_bound = [0 for _ in range(self.number_of_variables)]


    def update(self, *args, **kwargs):
       counter = kwargs["COUNTER"]
       self.time = (1.0 / self.nT) * math.floor(counter * 1.0 / self.tau)
       self.gt = math.sin(0.5 * math.pi * self.time)

       self.problem_modified = True

    def the_problem_has_changed(self):
        return self.problem_modified

    def clear_changed(self):
        self.problem_modified = False

    def evaluate(self, solution):
        pass

