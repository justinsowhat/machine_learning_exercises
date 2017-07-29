import numpy as np
import matplotlib.pyplot as plt
from nn_functions import *

Nclass = 500
D = 2
M = 3
K = 3

# randomly generating 3 guassian clouds
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)
N = len(Y)

T = np.zeros((N, K))
for i in xrange(N):
    T[i, Y[i]] = 1

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

learning_rate = 10e-7
costs = []
for epoch in xrange(10000):
    output, hidden = forward(X, W1, b1, W2, b2)
    if epoch % 100 == 0:
        c = cost(T, output)
        P = np.argmax(output, axis=1)
        r = accuracy(Y, P)
        print "cost: {0}\taccuracy: {1}".format(c, r)
        costs.append(c)

    W2 += learning_rate * derivative_w2(hidden, T, output)
    b2 += learning_rate * derivative_b2(T, output)
    W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
    b1 += learning_rate * derivative_b1(T, output, W2, hidden)

plt.plot(costs)
plt.show()
