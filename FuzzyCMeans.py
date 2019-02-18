import pandas as pd
import numpy as np
import random
import operator
import math
import sys

dataset = pd.read_csv(sys.argv[1])
columns = list(dataset.columns)
features = columns[:len(columns)]
datasetFeatures = dataset[features]

# Number of Clusters
k = int(sys.argv[2])

# Number of data points
n = len(datasetFeatures)

# Fuzzy parameter
fuzzyParam = 2.00


def memMatrixFunc():
    memMatrix = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        memMatrix.append(temp_list)
    return memMatrix


def locateCluster(memMatrix):
    cluster_mem_val = list(zip(*memMatrix))
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** fuzzyParam for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = list(datasetFeatures.iloc[i])
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def adjustMemMatrix(memMatrix, cluster_centers):
    p = float(2/(fuzzyParam-1))
    for i in range(n):
        x = list(datasetFeatures.iloc[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            memMatrix[i][j] = float(1/den)       
    return memMatrix


def clusterRetrieval(memMatrix):
    cluster_labels = list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(memMatrix[i]))
        cluster_labels.append(idx)
    return cluster_labels


def fuzzyCMeansInit():
    memMatrix = memMatrixFunc()
    curr = 0
    
    cluster_centers = locateCluster(memMatrix)
    memMatrix = adjustMemMatrix(memMatrix, cluster_centers)
    cluster_labels = clusterRetrieval(memMatrix)
    curr += 1
    return cluster_labels, cluster_centers




centroidsArr =[]
datapointLabel = []
ErrorArr =[]
for iter in range(int(sys.argv[3])):
    clustersOut,centroids = fuzzyCMeansInit()
    centroidsArr.append(centroids)
    datapointLabel.append(clustersOut)

for iterationInd, clustersOutDPLabel in enumerate(datapointLabel) :
    E = 0 
    for data in range(dataset.shape[0]):
        E += np.linalg.norm(dataset.values[data]-centroidsArr[0][datapointLabel[iterationInd][data]])
            
    ErrorArr.append(E)
    print("Final Error for Iteration: ",iterationInd," ",E)

minErrorCluster = datapointLabel[ErrorArr.index(min(ErrorArr))] # cluster with minimum error selected

np.savetxt(sys.argv[4], minErrorCluster, delimiter=',',fmt='%s') # save output in comma separated filename
