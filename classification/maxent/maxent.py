# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy as np
from math import exp
import scipy.misc
import matplotlib.pyplot as plt


class MaxEnt(Classifier):

    def __init__(self, model={}):
        super(MaxEnt, self).__init__(model)
        self.features = {}
        self.labels = {}
        self.parameters = np.zeros((1, 1))
        self.experiment_batch_size = False

    def get_model(self):
        return {'features': self.features, 'labels': self.labels, 'parameters': self.parameters}

    def set_model(self, model):
        if len(model) != 0:
            self.parameters = model['parameters']
            self.labels = model['labels']
            self.features = model['features']
        else:
            return

    model = property(get_model, set_model)

    def set_experiment_batch_size(self, bool):
        self.experiment_batch_size = bool

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""

        self.labels = {}
        self.features = {}

        self.generate_matrix_indices(instances)

        if dev_instances:
            self.generate_matrix_indices(dev_instances)
            self.generate_feature_vectors(dev_instances)

        self.generate_feature_vectors(instances)
        self.parameters = np.zeros((len(self.labels), len(self.features)))

        #collect data for experiment 1
        if self.experiment_batch_size:
            for entry in [(1, 'ro-'), (10, 'bs-'), (30, 'yo-'), (50, 'gs-'), (100, 'r^-'), (1000, 'b^-')]:
                batch_size, style = entry
                print("Training on batch size %s" % batch_size)
                self.parameters = np.zeros((len(self.labels), len(self.features)))
                results = self.train_sgd(instances, dev_instances, 0.0001, batch_size)
                x, y = self.dict_to_arrays(results)
                plt.plot(x, y, style)
            plt.axis([0, 150000, 60, 75])
            plt.ylabel("Accuracy")
            plt.xlabel("Number of Datapoints")
            plt.savefig('./batch_size_plot.png')
            plt.close()

        self.train_sgd(instances, dev_instances, 0.0001, 100)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient."""

        "initializing gradient and parameters with zeros, and an infinitive likelihood"
        gradient = np.zeros((len(self.labels), len(self.features)))
        current_parameters = np.copy(self.parameters)
        current_likelihood = float("inf")

        if dev_instances:
            evaluate_set = dev_instances
        else:
            evaluate_set = train_instances

        iteration = 0
        results = {}
        num_datapoints = 0
        accuracy = self.accuracy(evaluate_set)
        results[num_datapoints] = accuracy
        print "\nIteration %s ->> likelihood: %s \t accuracy: %s" % (iteration, current_likelihood, accuracy)

        converged = False
        while not converged:
            for i, instance in enumerate(train_instances):
                gradient += self.compute_gradient(instance)
                if i % batch_size == 0:
                    self.parameters += gradient * learning_rate
                if i % 1000 == 0:
                    num_datapoints += 1000
                    results[num_datapoints] = self.accuracy(evaluate_set)

            "checking if it's converged with the negative log likelihood"
            new_likelihood = self.compute_negative_loglikelihood(evaluate_set)
            iteration += 1
            print "Iteration %s ->> likelihood: %s \t accuracy: %s" % (iteration, new_likelihood, self.accuracy(evaluate_set))

            if new_likelihood < current_likelihood:
                np.copyto(current_parameters, self.parameters)
                current_likelihood = new_likelihood
                gradient = np.zeros((len(self.labels), len(self.features)))
            else:
                self.parameters = current_parameters
                converged = True
        print "Finish Training!"
        return results

    def classify(self, instance):
        scores = {}
        self.generate_feature_vector(instance)
        for label in self.labels.keys():
            index = self.labels[label]
            scores[label] = self.parameters[index].dot(instance.feature_vector)
        return max(scores, key=lambda l: scores[l])

    def accuracy(self, instances):
        nominator = sum([1 for instance in instances if self.classify(instance) == instance.label])
        return 100 * float(nominator) / len(instances)

    def compute_gradient(self, instance):
        gradient = np.zeros((len(self.labels), len(self.features)))
        "empirical values, which is basically counts"
        gradient[self.labels[instance.label]] += instance.feature_vector
        "gradient = empirical values - expected values"
        "expected value = posterior * feature_functions"
        for label in self.labels.keys():
            label_index = self.labels[label]
            gradient[label_index] -= self.compute_posterior(instance, label) * instance.feature_vector
        return gradient

    def compute_posterior(self, instance, label=None):
        if not label:
            label = instance.label
        unnormalized_score = self.parameters[self.labels[label]].dot(instance.feature_vector)
        unnormalized_score_sum = scipy.misc.logsumexp([self.parameters[index].dot(instance.feature_vector) for index in self.labels.values()])
        return exp(unnormalized_score - unnormalized_score_sum)

    def compute_negative_loglikelihood(self, instances, sigma=1.0):
        unnormalized_score = 0.0
        unnormalized_score_sum = 0.0
        for instance in instances:
            unnormalized_score += self.parameters[self.labels[instance.label]].dot(instance.feature_vector)
            unnormalized_score_sum += scipy.misc.logsumexp([self.parameters[index].dot(instance.feature_vector) for index in range(len(self.labels))])
        l2_regularizer = sum([((lamb ** 2)/(sigma ** 2)) for row in self.parameters for lamb in row])
        return - (unnormalized_score - unnormalized_score_sum - l2_regularizer)

    def generate_matrix_indices(self, instances):
        """indexing the labels and features for creating matrices"""
        for instance in instances:
            if instance.label not in self.labels:
                self.labels[instance.label] = len(self.labels)
            for feature in instance.features():
                if feature not in self.features:
                    self.features[feature] = len(self.features)

    def generate_feature_vector(self, instance):
        """using binary values to represent feature functions"""
        instance.feature_vector = np.zeros(len(self.features))
        for feature in instance.features():
            if feature not in self.features:
                continue
            instance.feature_vector[self.features[feature]] = 1

    def generate_feature_vectors(self, instances):
        """using binary values to represent feature functions"""
        for instance in instances:
            self.generate_feature_vector(instance)

    def dict_to_arrays(self, dict):
        keys = dict.keys()
        keys = sorted(keys)
        values = [dict[key] for key in keys]
        return keys, values

