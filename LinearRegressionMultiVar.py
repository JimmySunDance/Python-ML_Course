import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

plt.close('all')

def Hypoth(X, theta):
    return np.dot(X, theta)

def costFunction(X, y, theta):
    m = len(y)
    return (1./(2*m))*sum((Hypoth(X, theta) - y)**2)

def GradientDecMulti(X, y, theta, alpha, iterations):
    m = len(y)
    J_hist = np.zeros([iterations, 1])
    
    x1 = np.zeros([len(y), 1])
    x2 = np.zeros([len(y), 1])
    for i in range(len(y)):
        x1[i] = X[i, 1]
        x2[i] = X[i, 2]

    for i in range(iterations):
        
        t1 = float(theta[0]) - (alpha/m)*sum(Hypoth(X, theta) - y)
        t2 = float(theta[1]) - (alpha/m)*sum((Hypoth(X, theta) - y) * x1)
        t3 = float(theta[2]) - (alpha/m)*sum((Hypoth(X, theta) - y) * x2)
       
        theta = np.array([[float(t1)], [float(t2)], [float(t3)]])
        J_hist[i] = costFunction(X, y, theta)
        
    return theta, J_hist

def FeatureNorm(XMat):
    mu = np.mean(XMat)
    sigma = max(XMat)-min(XMat)
    return (XMat - mu)/ sigma
    
def NormalEqn(X, y):
    return np.dot(np.dot(inv(np.dot(np.transpose(X), X)), np.transpose(X)), y)

data = np.genfromtxt('ex1data2.txt')


m = len(data)


XMat = np.ones([m, int(np.shape(data)[1])])
yMat = np.ones([m, 1])
for i in range(m):
    XMat[i, 1] = data[i, 0]
    XMat[i, 2] = data[i, 1]
    yMat[i] = data[i, 2]

XMat[:, 1] = FeatureNorm(XMat[:, 1])
XMat[:, 2] = FeatureNorm(XMat[:, 2])

alpha = 0.01
iterations = 2000
theta = np.zeros([3, 1])

Theta, Jhist = GradientDecMulti(XMat, yMat, theta, alpha, iterations)


plt.figure()
plt.plot(Jhist)

plt.figure()
plt.plot(yMat)
plt.show()


price = np.dot(FeatureNorm(np.array([1,1650, 3])), NormalEqn(XMat, yMat))




