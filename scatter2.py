import sys
import numpy as np
import pandas as pd
from numpy import genfromtxt

if len(sys.argv) != 5:
  print(sys.argv[0], "takes 5 arguments. Not ", len(sys.argv)-1)
  sys.exit()

first = sys.argv[1]
second = sys.argv[2]

Xt = np.asmatrix( pd.read_csv(first, header=None).as_matrix() ) #read dataset
y = genfromtxt(second, delimiter=',')       #read label dataset

mu = np.mean(Xt, axis=0)            #mean of the dataset

n,m = Xt.shape                      #find shape of the input matrix

#initialize the individual scatter matrix of each group and within class scatter matrix 
W = np.zeros((m,m))     #initialize within class scatter matrix 
W1 = np.zeros((m,m))    #initialize scatter matrix of each group - label 1
W2 = np.zeros((m,m))    #initialize scatter matrix of each group - label 2
W3 = np.zeros((m,m))    #initialize scatter matrix of each group - label 3

mu1 = np.zeros((1,m))   #initialize mean matrix of group 1
mu2 = np.zeros((1,m))   #initialize mean matrix of group 2
mu3 = np.zeros((1,m))   #initialize mean matrix of group 3

count1=0                #number of vectors in group 1
count2=0                #number of vectors in group 2
count3=0                #number of vectors in group 3

for i, xt in enumerate(Xt) :    #iterate over each data point
    if y[i] == 1 :
        mu1+=xt
        count1+=1
    if y[i] == 2:
        mu2+=xt
        count2+=1
    if y[i] == 3:
        mu3+=xt
        count3+=1

mu1=mu1/count1              #mean matrix of group 1
mu2=mu2/count2              #mean matrix of group 2
mu3=mu3/count3              #mean matrix of group 3

for i, xt in enumerate(Xt) :
    if y[i] == 1 :
        W1 +=np.dot(((xt-mu1).T), (xt-mu1))         #scatter matrix of each group - label 1
    if y[i] == 2:
        W2 +=np.dot(((xt-mu2).T), (xt-mu2))         #scatter matrix of each group - label 2
    if y[i] == 3:
        W3 += np.dot(((xt-mu3).T), (xt-mu3))        #scatter matrix of each group - label 3

W=W1+W2+W3                              #within class scatter matrix = sum of scatter matrix of each group 

M = np.zeros((m,m))                     #initialize the mixture (total) scatter matrix

for i, xt in enumerate(Xt) :
    M+=np.dot(((xt-mu).T),(xt-mu))      # mixture (total) scatter matrix


B = np.zeros((m,m))                     #iniatize between class scatter matrix
B = M - W                               #between class scatter matrix as per Theorem: M = W + B

evals,evecs = np.linalg.eigh(B)  # Eigen Value and Vector of Between Class matrix

idx = np.argsort(evals)[::-1] # sort in descending order as we need to maximize
evals = evals[idx]
evecs = evecs[:,idx]

r = 2               # since we need to reduce the dimension to n*2
Vec_r = evecs[:,:r]  # get first r eigenvectors

feature = np.dot(Xt,Vec_r) #extract the feature 

np.savetxt(sys.argv[3], Vec_r.T, delimiter=',') #write eigen vectors into the file
np.savetxt(sys.argv[4], feature, delimiter=',') #write the extracted feature to the filename fetched from command line
