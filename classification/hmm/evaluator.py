"""Evaluator functions

You will have to implement more evaluator functions in this class.
We will keep them all in this file.
"""
import numpy


def compute_cm(classifier, test_data):
    """Evaluate the classifier given test data

    Evaluate the model on the test set and returns the evaluation
    result in a confusion matrix

    Returns:
        Confusion matrix
    """

    confusion_matrix = ConfusionMatrix(classifier.labels)
    for instance in test_data:
        prediction = classifier.classify(instance)
        confusion_matrix.add_data(prediction, instance.label)
    return confusion_matrix


class ConfusionMatrix(object):

    def __init__(self, labels):
        """
        initialize the confusion matrix.

        Args:
            label2index: a dictionary map from label string to its index
            index2label: a list map from label index to its string 
        """
        
        self.label_codebook = labels
        self.index2label = labels.index_to_label
        self.num_classes = labels.size
        self.matrix = numpy.zeros((self.num_classes, self.num_classes))
        
    def add_data(self, prediction_list, true_answer_list):
        for prediction, true_answer in zip(prediction_list, true_answer_list):
            self.matrix[int(prediction), self.label_codebook.get_index_by_label(true_answer)] += 1
            
    def compute_precision(self):
        """Returns a numpy.array where precision[i] = precision for class i""" 
        precision = numpy.zeros(self.num_classes)
        for i in range(0, self.num_classes):
            precision[i] = self.matrix[i, i]/self.matrix[i].sum()
        return precision
        
    def compute_recall(self):
        """Returns a numpy.array where recall[i] = recall for class i""" 
        recall = numpy.zeros(self.num_classes)
        for i in range(0, self.num_classes):
            recall[i] = self.matrix[i, i]/self.matrix.sum(0)[i]
        return recall
        
    def compute_f1(self):
        """Returns a numpy.array where f1[i] = F1 score for class i
        
        F1 score is a function of precision and recall, so you can feel free
        to call those two functions (or lazily load from an internal variable)
        But the confusion matrix is usually quite small, so you don't need to worry
        too much about avoiding redundant computation.
        """ 
        f1 = numpy.zeros(self.num_classes)
        precision = self.compute_precision()
        recall = self.compute_recall()
        f1 = 2*precision*recall/(precision+recall)
        return f1
        
    def compute_accuracy(self):
        """Returns accuracy rate given the information in the matrix"""
        correct_counts = 0.
        for i in range(0, self.num_classes):
            correct_counts += self.matrix[i][i]
        accuracy = correct_counts/self.matrix.sum()
        
        return accuracy
        
    def print_out(self):
        """Printing out confusion matrix along with Macro-F1 score"""
        # header for the confusion matrix
        header = [' '] + [self.index2label[i] for i in xrange(self.num_classes)]
        rows = []
        # putting labels to the first column of rhw matrix
        for i in xrange(self.num_classes):
            row = [self.index2label[i]] + [str(self.matrix[i,j]) for j in xrange(len(self.matrix[i,]))]
            rows.append(row)
        print '\n\nConfusion Matrix'
        print '--------------'
        print "row = predicted, column = truth"
        print matrix_to_string(rows, header)
        
        # computing precision, recall, and f1
        precision = self.compute_precision()
        recall = self.compute_recall()
        f1 = self.compute_f1()
        for i in xrange(self.num_classes):
            print '%s \tprecision %f \trecall %f\t F1 %f' % (self.index2label[i], 
                                                             precision[i], recall[i], f1[i])
        accuracy = self.compute_accuracy()
        print 'accuracy rate = %f' % accuracy
        return precision, recall, f1, accuracy


def matrix_to_string(matrix, header=None):
    """
    Return a pretty, aligned string representation of a nxm matrix.
    
    This representation can be used to print any tabular data, such as
    database results. It works by scanning the lengths of each element
    in each column, and determining the format string dynamically.
    
    the implementation is adapted from here
    mybravenewworld.wordpress.com/2010/09/19/print-tabular-data-nicely-using-python/
    
    Args:
    matrix - Matrix representation (list with n rows of m elements).
    header -  Optional tuple or list with header elements to be displayed.
    
    Returns:
    nicely formatted matrix string
    """
    
    if isinstance(header, list):
        header = tuple(header)
    lengths = []
    if header:
        lengths = [len(column) for column in header]
        
    # finding the max length of each column
    for row in matrix:
        for column in row:
            i = row.index(column)
            column = str(column)
            column_length = len(column)
            try:
                max_length = lengths[i]
                if column_length > max_length:
                    lengths[i] = column_length
            except IndexError:
                lengths.append(column_length)
                
    # use the lengths to derive a formatting string
    lengths = tuple(lengths)
    format_string = ""
    for length in lengths:
        format_string += "%-" + str(length) + "s "
    format_string += "\n"
    
    # applying formatting string to get matrix string
    matrix_str = ""
    if header:
        matrix_str += format_string % header
    for row in matrix:
        matrix_str += format_string % tuple(row)
        
    return matrix_str
    

