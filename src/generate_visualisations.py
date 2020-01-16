import sys

from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import *

def print_mean(datafile, indicator_name):
    avg = compute_mean_indicator(filename=datafile, indicator_name=indicator_name)
    print("{} avg:".format(indicator_name))
    print("{} \n".format(avg))

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

        print_mean(filename, "IGD")
        print_mean(filename, "HV")

    #generate_latex_tables(filename=datafile)

    # Generate boxplots
    #generate_boxplot(filename=datafile)

    # Wilcoxon
    #compute_wilcoxon(filename=datafile)

