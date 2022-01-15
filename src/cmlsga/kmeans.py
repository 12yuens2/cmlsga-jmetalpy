import numpy as np

from jmetal.config import store
from sklearn.cluster import KMeans
from sklearn import svm

def Clustering_kmeans(population, number_of_collectives):
    kmeans = KMeans(n_clusters=number_of_collectives, max_iter = 1000)

    #create the matrix of variables
    pop_temp = []
    for solution in population:
        pop_temp.append(solution.variables)

    return kmeans.fit_predict(pop_temp)+1


def Clustering(population, number_of_collectives, problem):
    labels = generate_labels(population, number_of_collectives)

    #generator = store.default_generator
    #train_pop = [generator.new(problem) for _ in range(len(population))]
    #clf = svm.SVC(decision_function_shape='ovr')
    #real_pop = [s.variables for s in population]
    #clf.fit([s.variables for s in train_pop], labels)
    #clf.predict(real_pop)

    return labels

def generate_labels(population, number_of_collectives):
    individual_average = [sum(s.variables) / len(population) for s in population]
    avg = sum(individual_average) / len(individual_average)
    sd = np.std(individual_average)

    #todo refactor
    labels = [0 for i in range(len(population))]
    if number_of_collectives == 4:
        for i in range(len(population)):
            if individual_average[i] < avg - 0.6*sd:
                labels[i] = 1
            elif individual_average[i]  < avg:
                labels[i] = 2
            elif individual_average[i]  < avg + 0.6*sd:
                labels[i] = 3
            else:
                labels[i] = 4

    elif number_of_collectives == 6:
        for i in range(len(population)):
            if individual_average[i] < avg - sd:
                labels[i] = 1
            elif individual_average[i]  < avg - sd / 2:
                labels[i] = 2
            elif individual_average[i]  < avg:
                labels[i] = 3
            elif individual_average[i]  < avg + sd / 2:
                labels[i] = 4
            elif individual_average[i]  < avg + sd:
                labels[i] = 5
            else:
                labels[i] = 6

    elif number_of_collectives == 8:
        for i in range(len(population)):
            if individual_average[i] < avg - 1.1*sd:
                labels[i] = 1
            elif individual_average[i]  < avg - 0.6*sd:
                labels[i] = 2
            elif individual_average[i]  < avg - 0.2*sd:
                labels[i] = 3
            elif individual_average[i]  < avg:
                labels[i] = 4
            elif individual_average[i]  < avg + 0.2*sd:
                labels[i] = 5
            elif individual_average[i]  < avg + 0.6*sd:
                labels[i] = 6
            elif individual_average[i]  < avg + 1.1*sd:
                labels[i] = 7
            else:
                labels[i] = 8

    else:
        print("Number of collectives: {} not supported.".format(number_of_collectives))
        exit()

    return labels
