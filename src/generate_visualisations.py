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
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    stat = compute_indicator(datafile, indicator_name, stat_function)
    print("{} {}:".format(indicator_name, stat_function))
    print("{} \n".format(stat))

def add_reference_front(indicator, reference_front_file):
    if Path(reference_front_file).is_file():
        reference_front = []
        with open(reference_front_file) as file:
            for line in file:
                reference_front.append([float(x) for x in line.split()])

        indicator.reference_front = reference_front
    else:
        print("no reference front for {}".format(reference_front_file))

def add_reference_point(indicator, reference_front_file):
    if Path(reference_front_file).is_file():
        with open(reference_front_file) as f:
            num_obj = len(f.readline().split())
            max_values = [-sys.float_info.min for i in range(num_obj)]
            values = []

            f.seek(0) # reset file to beginning
            for line in f:
                values.append([float(x) for x in line.split()])

            for i in range(num_obj):
                nadir = max([v[i] for v in values])
                max_values[i] = nadir + (nadir * 0.1) # 10% worse than nadir point

            indicator.referencePoint = max_values
    else:
        print("no reference front for ref point in {}".format(reference_front_file))


def generate_summary_from_experiment(input_dir, quality_indicators, evaluations,
                                     reference_fronts = ''):
    if not quality_indicators:
        quality_indicators = []
        
    #with open('QualityIndicatorSummary.csv', 'w+') as of:
    #    of.write('Algorithm,Problem,ExecutionId,Evaluations,IndicatorName,IndicatorValue\n')

    for dirname, _, filenames in os.walk(input_dir):
        print(dirname)
        for filename in sorted(filenames):
            try:
                # Linux filesystem
                algorithm, problem = dirname.split('/')[-2:]
            except ValueError:
                # Windows filesystem
                algorithm, problem = dirname.split('\\')[-2:]
            print("{}, {}".format(algorithm, problem))
            if 'FUN' in filename:
                solutions = read_solutions(os.path.join(dirname, filename))
                digits = [s for s in filename.split('.') if s.isdigit()]
                run_tag = digits[0]
                evaluation_tag = evaluations
                if len(digits) > 1:
                    evaluation_tag = digits[1]

                for indicator in quality_indicators:
                    reference_front_file = "resources/reference_front/{}.pf".format(problem) 
  
                    # Add reference front if any
                    if hasattr(indicator, 'reference_front'):
                        add_reference_front(indicator, reference_front_file)

                    # Add reference point if any
                    if hasattr(indicator, 'referencePoint'):
                        add_reference_point(indicator, reference_front_file)

                    result = indicator.compute([solutions[i].objectives for i in range(len(solutions))])

                    # Save quality indicator value to file
                    with open('corrections/summaries-nov-24/QualityIndicatorSummary-{}-{}.csv'.format(algorithm, problem), 'a+') as of:
                        of.write(','.join([algorithm, problem, str(run_tag), str(evaluation_tag), indicator.get_short_name(), str(result)]))
                        of.write('\n')

if __name__ == "__main__":

    data_directory = sys.argv[1]

    for data in data_directories:
        print(data)
        generate_summary_from_experiment(
            data,
            [InvertedGenerationalDistance(None)],
            reference_fronts="resources/reference_front"
        )

        filename = data[5:].replace("/", "")
        print("rename to " + filename)
        os.rename("QualityIndicatorSummary.csv", filename)
        #filename = data

        print("Filename: " + filename)

        print_stat(filename, "IGD", "mean")
        print_stat(filename, "IGD", "min")
        print_stat(filename, "IGD", "max")

    #generate_latex_tables(filename=datafile)

    # Generate boxplots
    #generate_boxplot(filename=datafile)

    # Wilcoxon
    #compute_wilcoxon(filename)
>>>>>>> 903a598d6e4c6f11d0653317fc7195713274915b

    generate_summary_from_experiment(
        data_directory,
        [InvertedGenerationalDistance(None), HyperVolume(None)],
        100000,
        reference_fronts="resources/reference_front"
    )
