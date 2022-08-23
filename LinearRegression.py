import numpy as np
from matplotlib import pyplot as plt

def hypothesis(X, theta):
    return X.dot(theta)

def cost_function(loss):
    m = len(loss)
    return  np.sum(loss ** 2) / (2 * m)

def gradient_decent(x, y, theta, alpha, m, numIterations):
    J_hist = []

    for i in range(0, numIterations):
        loss = hypothesis(x, theta) - y
        J_hist.append(cost_function(loss))

        # update theta
        theta -= (alpha/m) * np.dot(x.T, loss)

    return theta, J_hist


data = np.genfromtxt('data/data1.csv', delimiter=",")
X, y = data[:, 0], data[:, 1]

m, n = data.shape
XMat = np.ones([m, 2])
yMat = y

for i in range(m):
    XMat[i, 1] = X[i]


numIterations= 10000
alpha = 0.0005
theta = np.ones(n)


print("Cost for init params:")
print(cost_function(hypothesis(XMat, theta) - yMat))

theta, Jhist = gradient_decent(XMat, yMat, theta, alpha, XMat.shape[0], numIterations)
print(theta)


plt.figure(1)
plt.scatter(XMat[:,1], yMat, marker="+", c="r")
plt.plot(
    range(100), 
    theta[1]*range(100) + theta[0]
)

plt.figure(2)
plt.plot(Jhist)

plt.show()
