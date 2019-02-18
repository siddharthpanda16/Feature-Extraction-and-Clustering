import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 5:
  print(sys.argv[0], "takes 5 arguments. Not ", len(sys.argv)-1)
  sys.exit()

first = sys.argv[1]

A = np.asmatrix( pd.read_csv(first, header=None).as_matrix() ) #reac the input data set

mu = np.mean(A, axis=0) #mean of the matrix

Ac = A - mu  # mean subtraction
AcT= Ac.T   #Matrix Transpose

ActDot = np.dot(AcT,Ac) # dot product of At and A

evals,evecs = np.linalg.eigh(ActDot)  # Eigen Value and Vector

idx = np.argsort(evals)[::-1] # sort in reverse order
evals = evals[idx]          #eigen values
evecs = evecs[:,idx]       #eigen vectors

r = 2
V_r = evecs[:,:r]         # get first r eigenvectors

feature = np.dot(Ac,V_r)  #extract the feature with dot product of matrix and eigen vectors

np.savetxt(sys.argv[3], V_r.T, delimiter=',') #write eigen vectors into the file
np.savetxt(sys.argv[4], feature, delimiter=',') #write the extracted feature to the filename fetched from command line



