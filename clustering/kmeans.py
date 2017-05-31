import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances


class KMeans(object):
    def __init__(self, X, K, smart_init=False, seed=None):
        """
        :param X: dataset
        :param K: number of clusters
        :param smart_init: whether the model uses kmeans++ to initialize clusters
        :param seed: random seed for smart_init
        """
        self.X = X
        self.K = K
        self.N, self.D = X.shape
        self.clusters = None

        if not smart_init:  # initialize random centroids
            rand_indices = np.random.randint(0, self.N, self.K)
            self.centroids = self.X[rand_indices, :]
        else:  # use k-means++ smart initialization
            self.centroids = self.__smart_init(self.K, self.X, seed=seed)

    def train(self, max_iter=20, verbose=False):

        prev_clusters = None

        for i in range(max_iter):

            self.clusters = self.__assign_clusters()
            self.centroids = self.__revise_centroids()

            # break when clusters are not updated
            if prev_clusters is not None and (prev_clusters == self.clusters).all():
                break

            if prev_clusters is not None:
                num_changed = np.sum(prev_clusters != self.clusters)
                if verbose:
                    print('Iteration #{0}: {1:5d} elements changed their cluster assignment.'.format(i, num_changed))
            prev_clusters = self.clusters[:]

    def plot_clusters(self):
        # need to fix this
        plt.plot(self.clusters)
        plt.title("Clusters")
        plt.show()

    def __assign_clusters(self):
        distances = pairwise_distances(self.X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def __revise_centroids(self):
        new_centroids = []
        for i in range(self.K):
            centroid = self.X[self.clusters == i].mean(axis=0)
            new_centroids.append(centroid.A1)
        return np.array(new_centroids)

    def __smart_init(self, K, X, seed=None):
        if seed is not None:
            np.random.seed(seed)
        centroids = np.zeros((K, X.shape[1]))
        rand_indices = np.random.randint(X.shape[0])
        centroids[0] = X[rand_indices, :].toarray()

        squared_distances = pairwise_distances(X, centroids[0:1], metric='euclidean').flatten() ** 2
        for i in range(1, K):
            idx = np.random.choice(X.shape[0], 1, p=squared_distances / sum(squared_distances))
            centroids[i] = X[idx, :].toarray()
            squared_distances = np.min(pairwise_distances(X, centroids[0:i + 1], metric='euclidean') ** 2, axis=1)

        return centroids


