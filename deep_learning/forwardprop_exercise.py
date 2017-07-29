import numpy as np
import matplotlib.pyplot as plt
from nn_functions import *

Nclass = 500

# randomly generating 3 guassian clouds
X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
X = np.vstack([X1, X2, X3])

Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
plt.show()

D = 2
M = 3
K = 3

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

P_Y_given_X = forwardprop(X, [W1,W2], [b1,b2])
P = np.argmax(P_Y_given_X, axis = 1)

assert(len(P) == len(Y))

print "classification rate for randomly chosen weights: %s" % accuracy(Y, P)
