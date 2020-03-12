

class MSCLPSO(ParticleSwarmOptimization):

    def __init__(self,
                 problem,
                 swarm_size,
                 termination_criterion=store.default_termination_criteria,
                 swarm_generator=store.default_generator,
                 swarm_evaluator=store.default_evaluator):
        super(MSCLPSO, self).__init__(problem, swarm_size)

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator


    def create_initial_solutions(self):
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]


    def evaluate(self, swarm):
        return self.swarm_evaluator.evaluate(swarm, self.problem)


    def stopping_condition_is_met(self):
        return self.termination_criterion.is_met

    
    def initialize_velocity(self, swarm):
        pass

    def initialize_particle_best(self, swarm):
        pass

    def initialize_global_best(self, swarm):
        pass

    def update_velocity(self, swarm):
        pass

    def update_particle_best(self, swarm):
        pass

    def update_global_best(self, swarm):
        pass

    def update_position(self, swarm):
        pass

    def perturbation(self, swarm):
        return swarm


    def get_result(self):
        pass

    def get_name(self):
        return "MSCLPSO"
