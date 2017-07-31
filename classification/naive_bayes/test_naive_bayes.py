# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from corpus import Document, BlogsCorpus, NamesCorpus
from naive_bayes import NaiveBayes, HandCraftedFeatures
from collections import defaultdict

import sys
from random import shuffle, seed
from unittest import TestCase, main, skip

class EvenOdd(Document):
    def features(self):
        """Is the data even or odd?"""
        return [self.data % 2 == 0]

class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Name(Document):
    def features(self, letters="abcdefghijklmnopqrstuvwxyz"):
        """From NLTK's names_demo_features: first & last letters, how many
        of each letter, and which letters appear."""
        name = self.data
        return ([name[0].lower(), name[-1].lower()] +
                [name.lower().count(letter) for letter in letters] +
                [letter in name.lower() for letter in letters])


def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return sum(correct) / len(correct)


def f1(classifier, test, verbose=sys.stderr, method="average"):
    f1_scores = defaultdict(lambda: 0.0)
    label_count = defaultdict(lambda: 0)
    matrix = defaultdict(lambda: 0)
    for x in test:
        matrix[(classifier.classify(x), x.label)] += 1
        label_count[x.label] += 1

    for prediction in label_count.keys():
        "calculate the scores for each class"
        tp = 0.0
        fp = 0.0
        fn = 0.0
        for truth in label_count.keys():
            if prediction == truth:
                tp += matrix[(prediction, truth)]
            else:
                fp += matrix[(prediction, truth)]
                fn += matrix[(truth, prediction)]

        f1_score = 0.0
        if tp > 0:
            f1_score = 2 * tp / (2 * tp + fp + fn) * 100
        f1_scores[prediction] = f1_score

    f1_score = 0.0

    if method == "average":
        f1_score = sum(f1_scores.values())/len(f1_scores.keys())

    if method == "weighted":
        "weight the labels by their true instances"
        for label in label_count.keys():
            weight = label_count[label]/sum(label_count.values())
            f1_score += (weight * f1_scores[label])

    if verbose:
        # print ("Accuracy: %s" % (100 * float(tp) / sum(label_count.values())))
        print ("F1: %s" % f1_score)
    return f1


class NaiveBayesTest(TestCase):
    u"""Tests for the naïve Bayes classifier."""

    def test_even_odd_bernoulli(self):
        """Classify numbers as even or odd using Bernoulli\n"""
        classifier = NaiveBayes(method='b')
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)

    def test_even_odd_multinomial(self):
        """Classify numbers as even or odd using Multinomial\n"""
        classifier = NaiveBayes()
        classifier.train([EvenOdd(0, True), EvenOdd(1, False)])
        test = [EvenOdd(i, i % 2 == 0) for i in range(2, 1000)]
        self.assertEqual(accuracy(classifier, test), 1.0)

    def split_names_corpus(self, document_class=Name):
        """Split the names corpus into training and test sets\n"""
        names = NamesCorpus(document_class=document_class)
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:6000], names[6000:])

    def test_names_nltk_bernoulli(self):
        """Classify names using NLTK features using Bernoulli\n"""
        train, test = self.split_names_corpus()
        classifier = NaiveBayes(method='b')
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def test_names_nltk_multinomial(self):
        """Classify names using NLTK features using Multinomial\n"""
        train, test = self.split_names_corpus()
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.70)

    def split_blogs_corpus_imba(self, document_class):
        blogs = BlogsCorpus(document_class=document_class)
        imba_blogs = blogs.split_imbalance()
        return (imba_blogs[:1600], imba_blogs[1600:])

    def test_blogs_imba_bernoulli(self):
        """Classify imbalanced blog authors using bag-of-words with Bernoulli NB\n"""
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes(method='b')
        classifier.train(train)
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)

    def test_blogs_imba_multinomial(self):
        """Classify imbalanced blog authors using bag-of-words with Multinomial NB\n"""
        train, test = self.split_blogs_corpus_imba(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        # you don't need to pass this test
        self.assertGreater(accuracy(classifier, test), 0.1)

    def split_blogs_corpus(self, document_class):
        """Split the blog post corpus into training and test sets\n"""
        blogs = BlogsCorpus(document_class=document_class)
        self.assertEqual(len(blogs), 3232)
        seed(hash("blogs"))
        shuffle(blogs)
        return (blogs[:3000], blogs[3000:])

    def test_blogs_bag_bernoulli(self):
        """Classify blog authors using bag-of-words with Bernoulli NB\n"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes(method='b')
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_blogs_bag_multinomial(self):
        """Classify blog authors using bag-of-words with Multinomial NB\n"""
        train, test = self.split_blogs_corpus(BagOfWords)
        classifier = NaiveBayes()
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)

    def test_blogs_improved_bernoulli(self):
        """Classify blog authors using handcrafted features with Bernoulli NB\n"""
        train, test = self.split_blogs_corpus(HandCraftedFeatures)
        classifier = NaiveBayes(method='b')
        classifier.train(train)
        self.assertGreater(accuracy(classifier, test), 0.55)
        f1(classifier, test, method="weighted")

    def test_blogs_improved_multinomial(self):
        """Classify blog authors using handcrafted features with Multinomial NB\n"""
        train, test = self.split_blogs_corpus(HandCraftedFeatures)
        classifier = NaiveBayes()
        classifier.train(train)
        # classifier.save('./multinomial_BN.model')
        # classifier = NaiveBayes().load('./multinomial_BN.model')
        self.assertGreater(accuracy(classifier, test), 0.55)
        f1(classifier, test, method="weighted")


if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)
