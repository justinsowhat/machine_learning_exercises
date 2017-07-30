import numpy as np


def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2), Z


def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()


def derivative_w2(Z, T, Y):
    N, K = T.shape
    M = Z.shape[1]
    return Z.T.dot(T-Y)


def derivative_b2(T, Y):
    return (T-Y).sum(axis=0)


def derivative_w1(X, Z, T, Y, W2):
    N, D = X.shape
    M, K = W2.shape
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return X.T.dot(dZ)


def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1-Z)).sum(axis=0)


def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)


def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))


def accuracy(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total
