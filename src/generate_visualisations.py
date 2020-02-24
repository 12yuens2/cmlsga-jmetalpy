import sys
import pandas as pd

from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import *

def compute_indicator(filename, indicator_name, stat_function):
    """
    Adapted from jmetal.lab.experiment.compute_mean_indicator
    Changed so that it takes an statistic function instead of just the mean
    """
    df = pd.read_csv(filename, skipinitialspace=True)

    if len(set(df.columns.tolist())) != 5:
        raise Exception('Wrong number of columns')

    algorithms = pd.unique(df['Algorithm'])
    problems = pd.unique(df['Problem'])

    # We consider the quality indicator indicator_name
    data = df[df['IndicatorName'] == indicator_name]

    # Compute for each pair algorithm/problem the average of IndicatorValue
    average_values = np.zeros((problems.size, algorithms.size))
    j = 0
    for alg in algorithms:
        i = 0
        for pr in problems:
            average_values[i, j] = getattr(data['IndicatorValue'][np.logical_and(
                data['Algorithm'] == alg, data['Problem'] == pr)], stat_function)()
            i += 1
        j += 1

    # Generate dataFrame from average values and order columns by name
    df = pd.DataFrame(data=average_values, index=problems, columns=algorithms)
    df = df.reindex(df.columns, axis=1)

    return df


def print_stat(datafile, indicator_name, stat_function):
    stat = compute_indicator(datafile, indicator_name, stat_function)
    print("{} {}:".format(indicator_name, stat_function))
    print("{} \n".format(stat))


if __name__ == "__main__":

    data_directories = sys.argv[1:]

    for data in data_directories:
        print(data)
        generate_summary_from_experiment(
            data,
            [InvertedGenerationalDistance(), HyperVolume([1.0, 1.0])],
            reference_fronts="resources/reference_front"
        )

        filename = data[5:-1]
        os.rename("QualityIndicatorSummary.csv", filename)
        filename = data

        print_stat(filename, "IGD", "mean")
        print_stat(filename, "IGD", "min")
        print_stat(filename, "IGD", "max")

    #generate_latex_tables(filename=datafile)

    # Generate boxplots
    #generate_boxplot(filename=datafile)

    # Wilcoxon
    #compute_wilcoxon(filename=datafile)

