import numpy as np
import matplotlib.pyplot as plt

from data_processing import get_data
from nn_functions import *
from sklearn.utils import shuffle

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in xrange(N):
        ind[i, y[i]] = 1
    return ind

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

M = 5
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)
Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

train_costs = []
test_costs = []
learning_rate = 0.0001
for i in xrange(10000):
    pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
    train_costs.append(ctrain)
    test_costs.append(ctest)

    W2 -= learning_rate * Ztrain.T.dot(pYtrain - Ytrain_ind)
    b2 -= learning_rate * (pYtrain - Ytrain_ind).sum()
    dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1-Ztrain * Ztrain)
    W1 -= learning_rate * Xtrain.T.dot(dZ)
    b1 -= learning_rate * dZ.sum(axis=0)

    if i % 1000 == 0:
        print i, ctrain, ctest

print "Training accuracy: %s" % accuracy(Ytrain, predict(pYtrain))
print "Test accuracy: %s" % accuracy(Ytest, predict(pYtest))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()
