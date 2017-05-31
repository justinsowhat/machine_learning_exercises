from clustering.kmeans import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize


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

if __name__ == "__main__":
    main()
