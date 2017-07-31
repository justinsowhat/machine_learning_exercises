import numpy as np
from clustering.kmeans import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize



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
            self.centroids = self.X[rand_indices, :].toarray()
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


def visualize_document_clusters(raw, X, vectorizer, model, display_content=True):
    """borrowed the method from the cluster course on Coursera"""
    print('==========================================================')
    centroids = model.centroids

    # Visualize each cluster c
    for c in range(model.K):
        # Cluster heading
        print(),
        # Print top 10 words with largest TF-IDF weights in the cluster
        idx = centroids[c].argsort()[::-1]
        top_10_words = []
        for i in range(10):
            top_10_words.append(vectorizer.get_feature_names()[idx[i]])
        print('Cluster {0:d}: [{1}]'.format(c, ", ".join(top_10_words)))

        if display_content:
            # Compute distances from the centroid to all data points in the cluster,
            # and compute nearest neighbors of the centroids within the cluster.
            distances = pairwise_distances(X, centroids[c].reshape(1, -1), metric='euclidean').flatten()
            distances[model.clusters != c] = float('inf')  # remove non-members from consideration
            nearest_neighbors = distances.argsort()
            for i in range(8):
                text = ' '.join(raw[nearest_neighbors[i]].split(None, 25)[0:50])
                print('\n* {0:.5f}\n  {1:s}\n  {2:s}'.format(distances[nearest_neighbors[i]], text[:90], text[90:180]
                                                                if len(text) > 90 else ''))
            print('==========================================================')


def main():
    # just to run the algorithm with some actual data
    newsgroups_train = fetch_20newsgroups(subset='train', categories=['alt.atheism', 'talk.religion.misc',
                                                                      'comp.graphics', 'sci.space'])
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroups_train.data)
    X = normalize(X)
    model = KMeans(X, 4, smart_init=False)
    model.train(max_iter=50, verbose=True)
    visualize_document_clusters(newsgroups_train.data, X, vectorizer, model, False)

if __name__ == "__main__":
    main()
