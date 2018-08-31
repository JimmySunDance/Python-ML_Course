import numpy as np
from matplotlib import pyplot as plt
plt.close('all')


def Hypoth(X, theta):
    return np.dot(X, theta)

def costFunction(X, y, theta):
    m = len(y)
    return (1./(2*m))*sum((Hypoth(X, theta) - y)**2)

def GradientDec(X, y, theta, alpha, iterations):
    m = len(y)
    J_hist = np.zeros([iterations, 1])
    
    x1 = np.zeros([len(y), 1])
    for i in range(len(y)):
        x1[i] = X[i, 1]

    for i in range(iterations):
        
        t1 = float(theta[0]) - alpha*(1./m)*sum(Hypoth(X, theta) - y)
        t2 = float(theta[1]) - alpha*(1./m)*sum((Hypoth(X, theta) - y) * x1)
       
        theta = np.array([[float(t1)], [float(t2)]])
        J_hist[i] = costFunction(X, y, theta)
        
    return theta, J_hist



data = np.genfromtxt('ex1data1.txt')


X, y = data[:, 0], data[:, 1]
m = len(y)


# ########### Plotting the data ###############
# plt.figure()
# plt.plot(data[:, 0], data[:, 1], '+r')
# plt.xlim(4, 25)
# plt.ylim(-5, 25)
# plt.xlabel('Population of city (x10$^{4}$)')
# plt.ylabel('Profit (\\$x10$^{4}$)')
# plt.show()
# #############################################

XMat = np.ones([m, 2])
yMat = np.ones([m, 1])
for i in range(len(X)):
    XMat[i, 1] = X[i]
    yMat[i] = y[i]
    
theta = np.array([[-1], [2]])#np.zeros([2, 1])
iterations = 1500
alpha = 0.01

Theta, Jhist = GradientDec(XMat, yMat, theta, alpha, iterations)

plt.plot(np.sort(XMat[:, 1], 0), np.sort(np.dot(XMat, Theta), 0), lw = 1)

plt.figure(2)
plt.plot(Jhist)

plt.show()
