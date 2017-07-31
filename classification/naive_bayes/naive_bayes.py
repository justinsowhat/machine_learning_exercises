# -*- mode: Python; coding: utf-8 -*-

from __future__ import division
from classifier import Classifier
from collections import defaultdict
from corpus import Document
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import math

# global variables for the helper functions
# to work around the fact that cPickle cannot
# pickle lambda functions
denominator = 0
all_features = set()


class NaiveBayes(Classifier):
    u"""A naïve Bayes classifier."""

    def __init__(self, model={}, method="m"):
        """
        initialize a Naive Bayes classifier
        :param model: an empty dictionary by default -- a dictionary of probabilities
        :param method: 'm' for 'multinomial' and 'b' for 'bernoulli'
        """
        super(NaiveBayes, self).__init__(model)
        self.method = method

    def get_model(self):
        return self.seen

    def set_model(self, model):
        self.seen = model

    model = property(get_model, set_model)

    def train(self, instances, smoothing="additive"):
        """
        It trains a naive bayes model with given data set (instances)
        :param instances: data set with features
        :param smoothing: laplace, additive, and interpolation
        :return: nothing
        """
        # initialize most of the variables and counters
        probs = {}
        label_count = defaultdict(int)
        feature_count = defaultdict(set_default_dict)
        global denominator
        global p0
        global all_features

        # iterate over features
        for instance in instances:
            # for some reason, in the blogs data set there's a '' label????
            if instance.label == u'':
                continue
            label_count[instance.label] += 1
            if self.method == "m":
                # multinomial method
                for feature in instance.features():
                    feature_count[instance.label][feature] += 1
                    all_features.add(feature)
            else:
                # Bernoulli method
                for feature in set(instance.features()):
                    all_features.add(feature)
                    feature_count[instance.label][feature] += 1

        # Laplace smoothing
        if smoothing == "laplace":
            for label in label_count.keys():
                denominator = label_count[label] if (self.method == "b") else sum(feature_count[label].values())
                probs[label] = defaultdict(additive_default_prob)
                for feature in all_features:
                    count_x_y = feature_count[label][feature]
                    probs[label][feature] = additive_smoothing(count_x_y, denominator, 1, len(all_features))

        # Additive smoothing
        if smoothing == "additive":
            smoothing_factor = 0.2
            for label in label_count.keys():
                denominator = label_count[label] if (self.method == "b") else sum(feature_count[label].values())
                # probs[label] = defaultdict(lambda: additive_smoothing(0, denominator, smoothing_factor, len(all_features)))
                probs[label] = defaultdict(additive_default_prob)
                for feature in all_features:
                    count_x_y = feature_count[label][feature]
                    probs[label][feature] = additive_smoothing(count_x_y, denominator, smoothing_factor, len(all_features))

        # Jelinek-Mercer Smoothing or just simply interpolation
        # the 0th order model is the 1/feature_size
        # the 1st order model is count/sum_of_feature_counts
        if smoothing == "interpolation":
            # p0 = float(1)/len(all_features)
            l = 0.95
            for label in label_count.keys():
                denominator = label_count[label] if (self.method == "b") else sum(feature_count[label].values())
                # probs[label] = defaultdict(lambda: p0)
                probs[label] = defaultdict(interpolation_default_porb)
                for feature in all_features:
                    probs[label][feature] = l * feature_count[label][feature]/denominator + (1 - l) * p0

        for label in label_count.keys():
            probs[label]['**LABEL_PROBS**'] = float(label_count[label]) / sum(label_count.values())

        self.set_model(probs)

    def classify(self, instance):
        label_probs = {}
        for label in self.model.keys():
            label_probs[label] = math.log(self.model[label]['**LABEL_PROBS**'])
            if self.method == "b":
                test_features = set(instance.features())
                for feature in self.model[label]:
                    if feature == '**LABEL_PROBS**':
                        continue
                    if feature in test_features:
                        label_probs[label] += math.log(self.model[label][feature])
                    else:
                        label_probs[label] += math.log(1 - self.model[label][feature])
            else:
                for feature in instance.features():
                    label_probs[label] += math.log(self.model[label][feature])
        return max(label_probs, key=label_probs.get)


##########################
#   HELPER FUNCTIONS
#########################

def additive_smoothing(count_x_y, count_y, smoothing_factor, feature_size):
    return (count_x_y + smoothing_factor)/(count_y + smoothing_factor * feature_size)


def additive_default_prob():
    return additive_smoothing(0, denominator, 1, len(all_features))


def interpolation_default_porb():
    return float(1)/len(all_features)


def set_default_dict():
    return defaultdict(int)


######################################
#  FEATURE ENGINEERING
######################################


class HandCraftedFeatures(Document):
    def features(self):
        features = []

        # tokens = self.data.split()
        tokens = word_tokenize(self.data.lower())

        features += tokens
        # no_stopwords = [word for word in tokens if word not in stopwords.words('english')]

        # if len(self.data.split()) >= 500:
        #     features += ["**LONG BLOG POST**"]

        features += get_ngrams(tokens, 2)
        features += get_ngrams(tokens, 3)
        features += [word for word in tokens if word not in stopwords.words('english')]

        return features


def get_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

