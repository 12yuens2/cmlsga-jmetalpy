import sys

from functools import partial
from matplotlib import pyplot as plt

from jmetal.problem.multiobjective.dtlz import *
from jmetal.problem.multiobjective.zdt import *
from jmetal.problem.multiobjective.lz09 import *
from jmetal.problem.multiobjective.fda import *
from jmetal.lab.visualization import Plot, StreamingPlot
from jmetal.lab.visualization.streaming import pause
from jmetal.util.solution import read_solutions
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver

from cmlsga.algorithms.genetic_algorithms import *
from cmlsga.algorithms.particle_swarm_optimisation import *
from cmlsga.problems.uf import *
from cmlsga.problems.wfgopt import *
from cmlsga.problems.cdf import *
from cmlsga.problems.udf import *
from cmlsga.problems.fda import *


class MLSPlot(Plot):

    def __init__(self, **kwargs):
        super(MLSPlot, self).__init__(**kwargs)


    def two_dim(self, fronts, labels, filename, format):
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)

        reference = None
        if self.reference_front:
            reference, _ = self.get_points(self.reference_front)

        ax1 = None
        for i, _ in enumerate(fronts):
            points, _ = self.get_points(fronts[i])

            if ax1:
                ax = fig.add_subplot(n, n, i + 1, sharex = ax1, sharey = ax1)
            else:
                ax = fig.add_subplot(n, n, i + 1)
                ax1 = ax

            if self.reference_front:
                reference.plot(kind='scatter', x=0, y=1, s=3, ax=ax, color='k', legend=False)

            points.plot(kind='scatter', x=0, y=1, ax=ax, s=10, color='#236FA4', alpha=1.0)

            if labels:
                ax.set_title(labels[i])

            if self.axis_labels:
                plt.xlabel(self.axis_labels[0])
                plt.ylabel(self.axis_labels[1])

            ax.xaxis.set_tick_params(labelbottom=True)
            ax.yaxis.set_tick_params(labelleft=True)

            plt.setp(ax.get_xticklabels(), visible=True)
            plt.setp(ax.get_yticklabels(), visible=True)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=200)
        else:
            plt.show()

        plt.close(fig=fig)


from jmetal.core.observer import Observer

class MLSObserver(VisualizerObserver):

    def __init__(self, **kwargs):
        super(MLSObserver, self).__init__(**kwargs)

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        all_solutions = kwargs["ALL_SOLUTIONS"]
        #collectives = kwargs["COLLECTIVES"]

        if solutions:
            if self.figure is None:
                self.figure = MLSStreamingPlot(reference_point=self.reference_point,
                                            reference_front=self.reference_front)

                self.figure.plot(all_solutions, "all", color='y')
                self.figure.plot(solutions, "pf")

            if (evaluations % self.display_frequency) == 0:
                # check if reference point has changed
                reference_point = kwargs.get('REFERENCE_POINT', None)

                if reference_point:
                    self.reference_point = reference_point
                    self.figure.update(solutions, reference_point)
                else:
                    self.figure.update(all_solutions, "all")
                    self.figure.update(solutions, "pf")

                self.figure.ax.set_title('Eval: {}'.format(evaluations), fontsize=13)


class MLSStreamingPlot(StreamingPlot):

    def __init__(self, **kwargs):
        super(MLSStreamingPlot, self).__init__(**kwargs)

        self.sc = {}


    def plot(self, front, key, color="b"):
        # Get data
        points, dimension = Plot.get_points(front)

        # Create an empty figure
        self.create_layout(dimension)

        # If any reference point, plot
        if self.reference_point:
            for point in self.reference_point:
                self.scp, = self.ax.plot([[p] for p in point], c='r', ls='None', marker='', markersize=3)

        # If any reference front, plot
        if self.reference_front:
            rpoints, _ = Plot.get_points(self.reference_front)
            self.scf, = self.ax.plot(*[rpoints[column].tolist() for column in rpoints.columns.values],
                                     c='k', ls='None', marker='', markersize=1)

        # For MLS collectives
        #for collective in collectives:
        #    points, = Plot.get_points(collective.solutions)

        #    sc, = self.ax.plot(*[points[column].tolist() for column in points.columns.values],
        #                        ls='None', marker='o', markersize=4)
        #    self.sc.append(sc)


        # Plot data
        sc, = self.ax.plot(*[points[column].tolist() for column in points.columns.values],
                                c=color, ls='None', marker='o', markersize=4)
        self.sc[key] = sc

        # Show plot
        plt.show(block=False)


    def plot_collectives(self, front, collectives):
        # Get data
        points, dimension = Plot.get_points(front)

        # Create an empty figure
        self.create_layout(dimension)

        # If any reference point, plot
        if self.reference_point:
            for point in self.reference_point:
                self.scp, = self.ax.plot([[p] for p in point], c='r', ls='None', marker='', markersize=3)

        # If any reference front, plot
        if self.reference_front:
            rpoints,  = Plot.get_points(self.reference_front)
            self.scf, = self.ax.plot([rpoints[column].tolist() for column in rpoints.columns.values],
                                     c='k', ls='None', marker='', markersize=1)

        # For MLS collectives
        #for collective in collectives:
        #    points, = Plot.get_points(collective.solutions)

        #    sc, = self.ax.plot(*[points[column].tolist() for column in points.columns.values],
        #                        ls='None', marker='o', markersize=4)
        #    self.sc.append(sc)


        # Plot data
        self.sc, = self.ax.plot(*[points[column].tolist() for column in points.columns.values],
                                ls='None', marker='o', markersize=4)

        # Show plot
        plt.show(block=False)


    def update(self, front, key, reference_point=None):
        sc = self.sc[key]
        points, _ = Plot.get_points(front)
        sc.set_data(points[0], points[1])

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

        try:
            self.fig.canvas.flush_events()
        except KeyboardInterrupt:
            pass

        pause(0.01)


def run_algorithm(algorithm, problem, reference_front, population_size=20,
                  max_evaluations=100000, evaluator=store.default_evaluator):
    constructor, kwargs = algorithm(problem, population_size, max_evaluations, evaluator)
    algorithm = constructor(**kwargs)
    print(algorithm.get_name())
    algorithm.observable.register(ProgressBarObserver(max=max_evaluations))
    #algorithm.observable.register(MLSObserver(reference_front=reference_front, display_frequency=500))
    algorithm.run()

    return algorithm.get_result(), algorithm.get_name()

def run_plots(algorithms, problems):
    for problem in problems:
        print("Run {}".format(problem.get_name()))

        reference_front = read_solutions(
            filename="resources/reference_front/{}.pf".format(problem.get_name()))

        fronts, labels = zip(*[run_algorithm(algorithm, problem, reference_front)
                                for algorithm in algorithms])

        plot_front = MLSPlot(title="{} {} evaluations".format(problem.get_name(), 100000),
                        reference_front=reference_front, axis_labels=["x", "y"])

        plot_front.plot(list(fronts), label=list(labels),
                        normalize=True)
                        #filename="plots/{}.png".format(problem.get_name()))



#class Solution(object):
#    def __init__(self):
#        self.objectives = [0, 0, 0]
#        self.variables = [0.5] * 30
#        self.constraints = [0]
#
#s = Solution()
#
def write_pf(problem, time):
    filename = "{}_time{}.pf".format(problem.get_name(), time)
    points = problem.pf(1, 10000, time)

    with open(filename, "w") as f:
        for arr in points:
            for i in arr:
                f.write("{}\t".format(i))
            f.write("\n")
            f.flush()

    print(filename)

#reference_points = problem.pf(1, 5)
#print(list(reference_points))

#p = UF6()
for i in range(0, 41):
    write_pf(FDA1e(), i)
    write_pf(FDA2e(), i)
    write_pf(FDA3e(), i)
    write_pf(FDA4e(), i)
    write_pf(FDA5e(), i)
#    write_pf(FDA8(), i)

#    write_pf(CDF9(), i)
#    write_pf(CDF10(), i)

#write_pf(CDF11(), 1)

#run_plots([nsgaiii, omopso, cmpso, smpso], [ZDT4()])

#with open("100k-data.csv", "r") as f:
#    print(f.readline())
    

#from jmetal.lab.experiment import *
#
#compute_wilcoxon("100k-data.csv", "wilcoxon")
