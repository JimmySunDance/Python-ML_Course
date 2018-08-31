import numpy as np
import scipy.optimize as opt  
from matplotlib import pyplot as plt
  
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
 
def gradient(theta ,X, y):
    m, n = np.shape(X)
    grad = []
    for p in range(n):
        x1 = np.zeros([m, 1])    
        for i in range(len(y)):
            x1[i] = X[i, p]
        grad.append(float((sum(np.multiply((Hypoth(X, theta) - y),x1)))/m))
    return np.array(grad)
    
    
    
    
    
    
    


data = np.genfromtxt('ex2data1.txt')
 
m, n = data.shape
yMat = np.matrix(data[:, 2]).T
XMat = np.matrix([np.ones(m), data[:, 0], data[:, 1]]).T
 
ymat = np.array(data[:, 2]).T
Xmat = np.array([np.ones(m), data[:, 0], data[:, 1]]).T
 
 
 
# plt.figure()
# for i in range(m):
#     if yMat[i] == 0:
#         plt.plot(XMat[i, 1], XMat[i, 2], 'oy')
#     if yMat[i] == 1:
#         plt.plot(XMat[i, 1], XMat[i, 2], '+k')
# plt.xlim(30, 100), plt.ylim(30, 100)
# plt.xlabel('Test 1'), plt.ylabel('Test 2')
 
# plt.figure()
# plt.plot(np.linspace(-10, 10, 100), sigmoid(np.linspace(-10, 10, 100)))
 
 
 
 
 
 
theta_init = np.matrix([[0.], [0.], [0.]])
J = Cost(theta_init, XMat, yMat)
G = gradient(theta_init, XMat, yMat)
 
 
 
 
 
 
 
 
 
plt.show()