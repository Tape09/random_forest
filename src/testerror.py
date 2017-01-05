import numpy as np
import numpy.random as rnd
from scipy import stats
from Tree import *
from DataPool import *


class random_forest_test_error:  # {
    def __init__(self, data, data_type, y, y_type, n_classes, n_retry, number_of_trees, F, min_leaf_size=1, f_num='VR',
                 f_cat='IG'):  # {
        self.trees = []
        self.data = data
        self.y = y
        self.y_type = y_type

        indices = np.random.permutation(data.shape[0])
        idx = int(data.shape[0] * 0.9)
        training_idx, test_idx = indices[:idx], indices[idx:]
        training, test = data[training_idx, :], data[test_idx, :]
        for i in range(number_of_trees):
            new_tree = tree(training, data_type, y, y_type, n_classes, F, min_leaf_size, n_retry)
            self.trees.append(new_tree)

        self.test_data_index = test_idx

    # Test error
    def calculate_test_error(self, data, data_type, y, y_type, n_classes, n_retry, number_of_trees, F, min_leaf_size=1, f_num='VR',
                 f_cat='IG'):
        '''
        We separate the data in to training data and test data.
        Compute the test error
        In paper, repeating the test processing for 100 times and get the average value,
        every time we get the best performance in F = 1 and F =log2(M)+1.
        :return: the value of the test error
        '''
        error_rate_vote = 0
        for i in self.test_data_index:
            test_datum = data[i, :]
            true_value_test_data = y[i]
            for tree in self.trees:
                predictions = []
                # TODO: The prediction funciton can not predict a test datum
                # pred = tree.predict(test_datum)
                pred = 0
                predictions.append(pred)
            vote = predictions.count(true_value_test_data)/float(len(predictions))
            if vote < 0.5:
                error_rate_vote += 1
        error_rate = error_rate_vote/float(len(self.test_data_index))
        return error_rate


def test():
    data = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 3], [1, 4]])
    y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1])
    data_type = np.zeros(9)
    y_type = 1
    n_classes = np.array([2, 7])

    y0idx = y == 0
    y0data = data[y0idx, :]
    y1idx = y == 1
    y1data = data[y1idx, :]

    number_of_trees = 10
    F = 1
    n_retry = 4
    min_leaf_size = 5
    # new_tree = tree(data, data_type, y, y_type, n_classes, F, min_leaf_size, n_retry)

    rf_tt_er = random_forest_test_error(data, data_type, y, y_type, n_classes, n_retry, number_of_trees, F)
    error_rate = rf_tt_er.calculate_test_error(data, data_type, y, y_type, n_classes, n_retry, number_of_trees, F)
    print(error_rate)