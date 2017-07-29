import numpy as np

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forwardprop(X, weights, biases):
    assert len(weights) == len(biases)
    assert len(weights) > 1
    WB = list(zip(weights, biases))
    Z = np.tanh(X.dot(WB[0][0]) + WB[0][1])
    for w, b in WB[1:-1]:
        Z = np.tanh(Z.dot(w) + b)
    return softmax(Z.dot(WB[-1][0])+ WB[-1][1])

def accuracy(Y, P):
    n_correct = 0
    n_total = 0
    for i in xrange(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def classification_rate(Y, P):
    return np.mean(Y == P)
