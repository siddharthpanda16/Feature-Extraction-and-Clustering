import numpy as np
import random
import pandas as pd
import collections
import sys

def clusterAllocation(dataset, meanVal):
    clusters  = {}
    for x in dataset:
        clustersDict = min([(i[0], np.linalg.norm(x-meanVal[i[0]])) \
                    for i in enumerate(meanVal)], key=lambda t:t[1])[0]
        try:
            clusters[clustersDict].append(x)
        except KeyError:
            clusters[clustersDict] = [x]
    return clusters
 
def centreAdjustment(meanVal, clusters):
    updatedMean = []
    keys = sorted(clusters.keys())
    for k in keys:
        updatedMean.append(np.mean(clusters[k], axis = 0))
    return updatedMean
 
def convergenceCheck(meanVal, obsMean):
    return (set(np.asarray([tuple(a) for a in meanVal ]).ravel()) == set(np.asarray([tuple(a) for a in obsMean ]).ravel()))

def centreDistance(meanVal,dataset):
    D2 = np.array([min([np.linalg.norm(x-c)**2 for c in meanVal]) for x in dataset])
    return D2

def locateNextCenter(d,dataset):
    r = random.random()
    probs = d/d.sum()
    cumprobs =probs.cumsum()
    ind = np.where(cumprobs >= r)[0][0]
    return(dataset[ind])

def locateCentroids(dataset, K):
    # Initialize to K random centers
    obsMean = random.sample(dataset, K)
    meanVal = random.sample(dataset, K)
    while len(meanVal) < K:
        d = centreDistance(meanVal, dataset)
        meanVal.append(locateNextCenter(d, dataset))

    while not convergenceCheck(meanVal, obsMean):
        clusters = clusterAllocation(dataset, meanVal)
        obsMean = meanVal
        meanVal = centreAdjustment(obsMean, clusters)
    return(meanVal, clusters)

dataset = np.asarray( pd.read_csv(sys.argv[1], header=None).as_matrix() ) #read dataset
muArr =[]
clusterArr = []
ErrorArr =[]
for iter in range(int(sys.argv[3])):
    muOut,clustersOut = locateCentroids(list(dataset),int(sys.argv[2]))
    muArr.append(muOut)
    clusterArr.append(clustersOut)

for iterationInd, clustersOutDP in enumerate(clusterArr) :
    n,m = np.stack( clustersOutDP[0], axis=0 ).shape
    E =0  
    for i in range(len(clustersOutDP)):
        clusterMean=np.mean(np.stack( clustersOutDP[i], axis=0),axis=0 )
        clusterDatapoints=np.stack( clustersOutDP[i], axis=0 )

        for clusterInd, dataPoint in enumerate(clusterDatapoints) :
            E += np.linalg.norm(dataPoint-clusterMean)
            
    ErrorArr.append(E)
    print("Final Error for Iteration: ",iterationInd," ",E)

minErrorCluster = clusterArr[ErrorArr.index(min(ErrorArr))] # cluster with minimum error selected


clusterDatapoint = collections.defaultdict(list)
for key,values in minErrorCluster.items():
    for value in values:
        if not np.array2string(value,separator=',',suppress_small=True) in clusterDatapoint:
            clusterDatapoint[np.array2string(value,separator=',',suppress_small=True)].append(key)

clusterResult=[]
for x in dataset:
    clusterResult.append(clusterDatapoint[np.array2string(x,separator=',',suppress_small=True)])

np.savetxt(sys.argv[4], clusterResult, delimiter=',',fmt='%s') # save output in comma separated filename
