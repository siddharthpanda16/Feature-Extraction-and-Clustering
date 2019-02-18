import sys
import numpy as np
import pandas as pd
from numpy import genfromtxt

if len(sys.argv) != 7:
  print('usage : ', sys.argv[0], 'reduced_data_file vector_file labels_file queried_point_file label output_file')
  sys.exit()

reduced_file = sys.argv[1]
v_file = sys.argv[2]
labels_file = sys.argv[3]
queried_point_file = sys.argv[4]

A = np.asmatrix( pd.read_csv(reduced_file, header=None).as_matrix() )          #reduced data using dimension reduction tecnique n*2
V = np.asmatrix( pd.read_csv(v_file, header=None).as_matrix() )                #Eigen vectors from training data 2*m
Q = np.asmatrix( pd.read_csv(queried_point_file, header=None).as_matrix() )    #points to be tested n*m
Y = genfromtxt(labels_file, delimiter=',')                                     #label dataset
y = genfromtxt(sys.argv[5],dtype=int, delimiter=',')       #label value - 1,2,3 to find point nearest to Q of same label else the nearest point irrespective of label

if(y.size == 1) :         #when there is one element in y : make it an array.
  y = np.array([y])

reduced = np.dot(Q,V.T)                   #reduce the query point data - Q to n*2

minimum_dist = float("inf")               #set minimum as positive infinite
n,m = A.shape

nearest_nbour_arr = np.array( np.zeros((1,2)))  #initialise an array of 1*2 matrices for storing nearest neighbours point
index = np.zeros((0,1),dtype=str) #initialize the index of the nearest neighbours
euclid_distances = np.zeros((0,2))  #initialise a matrix to store index and euclidean distance of each data point

for dataIndex,dataRow in enumerate(Q):      #iterate over each data point to be queried from query file
  indexStr = ""  #index of nearest neighbours for each data point
  if(y[dataIndex] == 1 or y[dataIndex] == 2 or y[dataIndex] == 3) :          #if label is one of 1,2,3 then find nearest point of same label
    for i, xt in enumerate(A) :             #enumerate over each data point
      if Y[i] == y[dataIndex] :                        #filter  out the label not matching the input label
        euclid_dist = np.linalg.norm(xt-reduced[dataIndex]) # find euclidean distance of each data point from queried point
        euclid_distances = np.append(euclid_distances, np.array([[i,euclid_dist]]), axis=0) #store index and euclidean distance of each data point 
        if(euclid_dist < minimum_dist) :
          minimum_dist = euclid_dist        #store the minimum distance to the variable
    for i, row_euclid_dist in enumerate(euclid_distances) : #check for multiple nearest neighours for each data point
      if(row_euclid_dist[1]==minimum_dist) :
        indexStr += str(int(row_euclid_dist[0]))+"," #append all the nearest neighours for a point
 
  else :                                    # if label is not 1 or 2 or 3 find the nearest neighbour irrespective of point
    for i, xt in enumerate(A) :
      euclid_dist = np.linalg.norm(xt-reduced[dataIndex])
      euclid_distances = np.append(euclid_distances, np.array([[i,euclid_dist]]), axis=0)
      
      if(euclid_dist < minimum_dist) :
        minimum_dist = euclid_dist
    for i, row_euclid_dist in enumerate(euclid_distances) :
      if(row_euclid_dist[1]==minimum_dist) :
        indexStr += str(int(row_euclid_dist[0]))+","  #append all the nearest neighours for a point
  index = np.append(index,indexStr[:-1])  #append nearest neighbour indexes for each datapoint
  
print("minimum_dist : ", minimum_dist)
print("index of nearest neighbour : ", index)

output_file = sys.argv[6] # Output file name should be read from command line.
np.savetxt(output_file, index, delimiter=',',fmt='%s') # save output in comma separated filename
