from sklearn.cluster import KMeans

def Clustering(population, number_of_collectives):
    kmeans = KMeans(n_clusters=number_of_collectives, max_iter = 1000)

    #create the matrix of variables
    pop_temp = []
    for solution in population:
        pop_temp.append(solution.variables)


    return kmeans.fit_predict(pop_temp)+1
