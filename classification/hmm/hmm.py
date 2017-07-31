"""Hidden Markov Model sequence tagger

"""
from classifier import Classifier
import numpy
from utils import LabelIndexDict

SMOOTHING_FACTOR = 1


class HMM(Classifier):

    def __init__(self):
        super(HMM, self).__init__()
        # keeps track of 'BIO' label indices
        self.labels = LabelIndexDict()
        # keeps track of feature indices
        self.features = LabelIndexDict()

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def collect_labels(self, instance_list):
        """
        Build maps between labels and indices
        :param instance_list: the corpus
        :return: None
        """
        for instance in instance_list:
            for i in range(len(instance.label)):
                for j in range(len(instance.data[i])):
                    self.features.add(instance.data[i][j])
                    self.features.add(instance.data[i][j])
                self.labels.add(instance.label[i])

    def _collect_counts(self, instance_list):
        """Collect counts necessary for fitting parameters

        This function should update self.transtion_count_table
        and self.feature_count_table based on this new given instance

        Add your docstring here explaining how you implement this function

        Returns None"""
        self.initial_label_counts = numpy.zeros(self.labels.size)
        self.termination_label_counts = numpy.zeros(self.labels.size)
        self.transition_count_table = numpy.zeros((self.labels.size, self.labels.size))
        self.feature_count_table = numpy.zeros((self.features.size + 1, self.labels.size))
        for instance in instance_list:
            for i in range(len(instance.label)):
                label_index = self.labels.get_index_by_label(instance.label[i])
                if i == (len(instance.label) - 1):
                    self.termination_label_counts[label_index] += 1
                else:
                    if i == 0:
                        self.initial_label_counts[label_index] += 1
                    next_label_index = self.labels.get_index_by_label(instance.label[i + 1])
                    self.transition_count_table[next_label_index, label_index] += 1
                for j in range(len(instance.data[i])):
                    self.feature_count_table[self.features.get_index_by_label(instance.data[i][j]), label_index] += 1

        self._aggregate_low_count_words(0)

        self.initial_probabilities = self.initial_label_counts / self.initial_label_counts.sum()
        self.termination_probabilities = self.termination_label_counts / self.termination_label_counts.sum()

    def _aggregate_low_count_words(self, min_count=3):
        # aggregate the counts of words that are seen less than MIN_COUNT times to the last index, i.e. unknown word
        i = 0
        while i < self.features.size:
            if self.feature_count_table[i].sum() <= min_count:
                j = 0
                while j < self.labels.size:
                    self.feature_count_table[self.features.size, j] += self.feature_count_table[i, j]
                    j += 1
                label = self.features.get_label_by_index(i)
                self.features.set_index(label, self.features.size)
            i += 1

    def train(self, instance_list):
        """Fit parameters for hidden markov model

        Update codebooks from the given data to be consistent with
        the probability tables

        Transition matrix and emission probability matrix
        will then be populated with the maximum likelihood estimate
        of the appropriate parameters

        Add your docstring here explaining how you implement this function

        Returns None
        """
        self._collect_counts(instance_list)
        # plus one smoothing performs better than marking low frequency words as unknown
        self.transition_count_table += SMOOTHING_FACTOR
        self.feature_count_table += SMOOTHING_FACTOR
        self.transition_matrix = self.transition_count_table / numpy.sum(self.transition_count_table, axis=0)
        self.emission_matrix = self.feature_count_table / numpy.sum(self.feature_count_table, axis=0)

    def classify(self, instance):
        """Viterbi decoding algorithm

        Wrapper for running the Viterbi algorithm
        We can then obtain the best sequence of labels from the backtrace pointers matrix

        Add your docstring here explaining how you implement this function

        Returns a list of labels e.g. ['B','I','O','O','B']
        """
        best_sequence = []
        trellis, backtrace_pointers = self.dynamic_programming_on_trellis(instance, False)
        last_sequence = numpy.argmax(trellis[len(instance.data) - 1])
        best_sequence.insert(0, last_sequence)
        for i in reversed(range(1, len(instance.data))):
            best_sequence.insert(0, backtrace_pointers[i][last_sequence])
            last_sequence = backtrace_pointers[i][last_sequence]
        return best_sequence

    def compute_observation_loglikelihood(self, instance):
        """Compute and return log P(X|parameters) = loglikelihood of observations"""
        trellis = self.dynamic_programming_on_trellis(instance, True)
        return numpy.log(trellis[len(instance.data) - 1].sum())

    def dynamic_programming_on_trellis(self, instance, run_forward_alg=True):
        """Run Forward algorithm or Viterbi algorithm

        This function uses the trellis to implement dynamic
        programming algorithm for obtaining the best sequence
        of labels given the observations

        Add your docstring here explaining how you implement this function

        Returns trellis filled up with the forward probabilities
        and backtrace pointers for finding the best sequence
        """
        # Initialize trellis and backtrace pointers
        trellis = numpy.zeros((len(instance.data), self.labels.size))
        backtrace_pointers = numpy.zeros((len(instance.data), self.labels.size))
        # initialize the observation probabilities with features
        observation_probabilities = self._calculate_observation_probabilities(instance.data[0])
        trellis[0] = self.initial_probabilities * observation_probabilities

        for i in range(1, len(instance.data)):
            observation_probabilities = self._calculate_observation_probabilities(instance.data[i])
            if run_forward_alg:
                trellis[i] = (trellis[i - 1] * self.transition_matrix).sum(axis=1) * observation_probabilities
            else:
                forward_probabilities = trellis[i - 1] * self.transition_matrix
                trellis[i] = forward_probabilities.max(axis=1) * observation_probabilities
                backtrace_pointers[i] = numpy.argmax(forward_probabilities, axis=1)

        if run_forward_alg:
            return trellis
        else:
            return trellis, backtrace_pointers

    def train_semisupervised(self, unlabeled_instance_list, labeled_instance_list=None):
        """Baum-Welch algorithm for fitting HMM from unlabeled data (EXTRA CREDIT)

        The algorithm first initializes the model with the labeled data if given.
        The model is initialized randomly otherwise. Then it runs
        Baum-Welch algorithm to enhance the model with more data.

        Add your docstring here explaining how you implement this function

        Returns None
        """
        it = 0
        iterations = 10
        old_likelihood = 0.0
        self.initial_label_counts = numpy.zeros(self.labels.size)
        self.termination_label_counts = numpy.zeros(self.labels.size)

        if labeled_instance_list is not None:
            self.train(labeled_instance_list)
        else:
            # initialize the model randomly with floats between 0 and 1.0
            self.initial_probabilities = numpy.random.rand(self.labels.size)
            self.termination_probabilities = numpy.random.rand(self.labels.size)
            self.transition_matrix = numpy.random.rand(self.labels.size, self.labels.size)
            self.emission_matrix = numpy.random.rand(self.features.size + 1, self.labels.size)
        while it < iterations:
            #E-Step
            likelihood = 0.0
            self.expected_transition_counts = numpy.zeros((self.labels.size, self.labels.size))
            self.expected_feature_counts = numpy.zeros((self.features.size + 1, self.labels.size))
            for instance in unlabeled_instance_list:
                alpha_table, beta_table = self._run_forward_backward(instance)
                probability = (alpha_table[len(instance.data)-1] * self.termination_probabilities).sum()
                gamma = alpha_table * beta_table / probability
                for j in range(len(instance.data)):
                    next_observation_probabilities = numpy.zeros(self.labels.size)
                    for k in range(len(instance.data[j])):
                        # update the feature counts with counts from the gamma table
                        self.expected_feature_counts[self.features.get_index_by_label(instance.data[j][k]), :] += gamma[j]
                        if j < len(instance.data) - 1:
                            next_observation_probabilities += self.emission_matrix[self.features.get_index_by_label(instance.data[j + 1][k]), :]

                    if j < len(instance.data) - 1:
                        # update the transition counts with xi values
                        self.expected_transition_counts += alpha_table[j] * self.transition_matrix * next_observation_probabilities * beta_table[j + 1] / probability

                # add up the loglikelihood
                likelihood += numpy.log(probability)
                # update the count tables based on the estimated counts from the gamma table
                self.initial_label_counts += gamma[0]
                self.termination_label_counts += gamma[len(instance.data) - 1]

            self._aggregate_low_count_words(0)

            #M-Step
            self.initial_probabilities = self.initial_label_counts / self.initial_label_counts.sum()
            self.termination_probabilities = self.termination_label_counts / self.termination_label_counts.sum()
            self.transition_matrix = self.transition_count_table / numpy.sum(self.transition_count_table, axis=0)
            self.emission_matrix = self.feature_count_table / numpy.sum(self.feature_count_table, axis=0)

            print("Iteration #%s" % it + ", likelihood: %s" % likelihood)
            if self._has_converged(old_likelihood, likelihood):
                print("The algorithm has converged... Breaking...")
                break
            it += 1
            old_likelihood = likelihood

    def _has_converged(self, old_likelihood, likelihood):
        """Determine whether the parameters have converged or not (EXTRA CREDIT)

        Returns True if the parameters have converged.
        """
        return abs(old_likelihood - likelihood) < 0.0001

    def _run_forward_backward(self, instance):
        """Forward-backward algorithm for HMM using trellis (EXTRA CREDIT)

        Fill up the alpha and beta trellises (the same notation as
        presented in the lecture and Martin and Jurafsky)
        You can reuse your forward algorithm here

        return a tuple of tables consisting of alpha and beta tables
        """
        alpha_table = self._calculate_alpha_table(instance)
        beta_table = self._calculate_beta_table(instance)
        return alpha_table, beta_table

    def _calculate_observation_probabilities(self, instance_data):
        observation_probabilities = numpy.zeros(self.labels.size)
        for i in range(len(instance_data)):
            observation_probabilities += self.emission_matrix[self.features.get_index_by_label(instance_data[i]), :]
        return observation_probabilities

    def _calculate_alpha_table(self, instance):
        alpha_table = numpy.zeros((len(instance.data), self.labels.size))
        observation_probabilities = self._calculate_observation_probabilities(instance.data[0])
        alpha_table[0] = self.initial_probabilities * observation_probabilities

        # calculate the forward table
        for i in range(1, len(instance.data)):
            observation_probabilities = self._calculate_observation_probabilities(instance.data[i])
            alpha_table[i] = observation_probabilities * (alpha_table[i - 1] * self.transition_matrix).sum(axis=1)
        return alpha_table

    def _calculate_beta_table(self, instance):
        beta_table = numpy.zeros((len(instance.data), self.labels.size))
        beta_table[len(instance.data) - 1] = self.termination_probabilities

        # calculate the backward table
        for i in reversed(range(1, len(instance.data))):
            observation_probabilities = self._calculate_observation_probabilities(instance.data[i])
            beta_table[i - 1] = (beta_table[i] * self.transition_matrix * observation_probabilities).sum(axis=1)
        return beta_table