from corpus import Document, NamesCorpus, ReviewCorpus
from maxent import MaxEnt
from unittest import TestCase, main
from random import shuffle, seed
from nltk.corpus import stopwords
import sys
import re, string
import matplotlib.pyplot as plt


class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split() + ['***BIAS***']

class Name(Document):
    def features(self):
        name = self.data
        return ['First=%s' % name[0], 'Last=%s' % name[-1]] + ['***BIAS***']

class ReviewFeatures(Document):
    def features(self):
        features = ['***BIAS***']
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', self.data).split()
        stops = set(stopwords.words('english'))
        # remove negative words from the stop word list, since they should be strong features in sentiment analysis
        negative_words = ['not', 'no', 'nor', 'neither', 'aren', 'couldn', 'shouldn', 'weren', 'wasn', 'isn', 'don',
                          'didn', 'doesn', 'against', 'hasn', 'hadn', 'any', 'few', 'ain']
        stops = stops - set(negative_words)
        features += [token for token in text if token not in stops]
        return features

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return float(sum(correct)) / len(correct)

class MaxEntTest(TestCase):
    u"""Tests for the MaxEnt classifier."""

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training, dev, and test sets"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:5000], names[5000:6000], names[6000:])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        train, dev, test = self.split_names_corpus()
        classifier = MaxEnt()
        classifier.train(train, dev)
        acc = accuracy(classifier, test)
        self.assertGreater(acc, 0.70)

    def split_review_corpus(self, document_class, training_size=10000):
        """Split the yelp review corpus into training, dev, and test sets"""
        reviews = ReviewCorpus('yelp_reviews.json', document_class=document_class)
        seed(hash("reviews"))
        shuffle(reviews)
        dev_size = 1000
        test_size = 3000
        return (reviews[:training_size],
                reviews[training_size:training_size+dev_size],
                reviews[training_size+dev_size:training_size+dev_size+test_size])

    # def test_reviews_bag(self):
    #     """Classify sentiment using bag-of-words"""
    #     train, dev, test = self.split_review_corpus(BagOfWords)
    #     classifier = MaxEnt()
    #     classifier.train(train, dev)
    #     self.assertGreater(accuracy(classifier, test), 0.55)

    def test_reviews_test_batch_size(self):
        """Uses the handcrafted features to classify"""
        """Also plot the datapoints for batch size"""
        train, dev, test = self.split_review_corpus(ReviewFeatures)
        classifier = MaxEnt()
        classifier.set_experiment_batch_size(True)
        classifier.train(train, dev)
        classifier.save('QishenSu.Maxent.model')
        classifier = MaxEnt()
        classifier.load('QishenSu.Maxent.model')
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_reviews_test_train_size(self):
        # collecting data for experiment 2
        train_sizes = [1000, 3000, 5000, 8000, 10000, 20000, 30000, 50000, 100000]
        accuracies = []
        for size in train_sizes:
            print("Training model with training data of size %s" % size)
            train, dev, test = self.split_review_corpus(ReviewFeatures, size)
            classifier = MaxEnt()
            classifier.train(train, dev)
            accuracies.append(accuracy(classifier, test) * 100)
        plt.plot(train_sizes, accuracies, 'ro')
        plt.ylabel("Accuracy")
        plt.xlabel("Training Size")
        plt.savefig('./train_size_plot.png')
        plt.close()

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

