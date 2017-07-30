import numpy as np
import pandas as pd


def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()

    X = data[:, :-1]
    Y = data[:, -1]

    # normalize the numerical values
    X[:,1] = (X[:, 1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:, 2] - X[:,2].mean()) / X[:,2].std()

    N, D = X.shape
    X2 = np.zeros((N, D+3))
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]

    # convert cateogrical data to one-hot vectors
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1

    return X2, Y


def get_binary_data():
    # only return data from the first two classes
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]
    return X2, Y2
