import numpy as np
# import time
# import random as rnd
from matplotlib import pyplot as plt
import scipy.io as sio

  
plt.close('all')
 
 
def Hypoth(X, theta):
    return 1./(1+np.exp(-np.dot(X, theta)))
     
     
def Cost(theta, X, y):
    m = float(len(y))
    h = Hypoth(X, theta)
    term1 = np.multiply(-y, np.log(h))
    term2 = np.multiply((1-y), np.log(1-h))
    J = sum(term1-term2)/m
    return J
 
 
MAT = sio.loadmat('ex3data1.mat')


# XMat = MAT['X']
# yMat = MAT['y']
# 
# J = Cost()