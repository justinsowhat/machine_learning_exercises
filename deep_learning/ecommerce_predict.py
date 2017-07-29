import numpy as np
from data_processing import get_data
from nn_functions import *

X, Y = get_data()

M = 5
D = X.shape[1]
K = len(set(Y))
W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

P_Y_given_X = forwardprop(X, [W1,W2], [b1, b2])
predictions = np.argmax(P_Y_given_X, axis=1)

print "accuracy is %s" % classification_rate(P_Y_given_X, predictions)
