import sys

from jmetal.lab.experiment import *

def print_mean(datafile, indicator_name):
    avg = compute_mean_indicator(filename=datafile, indicator_name=indicator_name)
    print("{} avg:".format(indicator_name))
    print("{} \n".format(avg))

if __name__ == "__main__":

    datafiles = sys.argv[1:]

    for data in datafiles:
        print(data)

        print_mean(data, "IGD")
        print_mean(data, "HV")

    #generate_latex_tables(filename=datafile)

    # Generate boxplots
    #generate_boxplot(filename=datafile)

    # Wilcoxon
    #compute_wilcoxon(filename=datafile)

