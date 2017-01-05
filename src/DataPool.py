#! /usr/bin/env python
import numpy as np
import os

class DataPool:
    """
    The collection of all the data
    Take use of the methods to retrieve the data we need
    """
    def __init__(self, name_data):
        # only for glass
        self.name_data = name_data
        self.datapath = self.__get_paths(name_data)
        self.rawdata = self.__readdata()  # without any processing
        self.data = np.array(self.rawdata[:, 1:-1],dtype=np.float)  # only the data

        [self.num_samples, self.num_features] = self.data.shape
        self.class_v = np.array(self.rawdata[:, -1],dtype=np.int)
        self.attribute_type = [0] * 9  # 0: numerical, 1: categorical
        self.cla_reg = 1  # Whether classification or regression, 1 is classification.
        self.num_class = 6

    # data_type, y, y_type, n_classes, min_leaf_size = 1, n_retry = 1,
    def __get_paths(self, data_name):
        '''
        The directory system:
        The project is in the file named "project"
        It has "src", which save all the source codes, and "data", which has all the training data.

        Get the project path and training data path
        :param data_name: the name of the test data
        :return: training data path
        '''
        src_path = os.getcwd()  # get the directory of src
        project_path = os.path.dirname(src_path)  # get the parent directory of src, that is project
        data_path = os.path.join(project_path, 'data')  # get the directory of data, which should be in the project
        training_data_dir = os.path.join(data_path, data_name)  # now we are in the folder containing the training data
        return training_data_dir

    def __readdata(self):
        '''
        The directory system:
        The project is in the file named "project"
        It has "src", which save all the source codes, and "data", which has all the training data.

        Data are separated with ','
        Read the data into the a 'matrix' which is ndarray in python
        indexing in this way
        data[1,2], data[:,-1] or data[1,:]

        Sometimes, the first column is the index of the data
        Somtimes, the last column is the class of the data, or the first column
        :param datapath: the path of the data
        :return: 'data' is ndarray
        '''
        data_l = []
        with open(self.datapath, "r") as f:
            mylist = f.read().splitlines()
            for line in mylist:
                currentline = line.split(",")
                data_l.append(currentline)
        data = np.array(data_l)
        return data


def demo():
    data = DataPool('glass.txt')
    print (data.data)
    print (data.class_v)
    print (data.attribute_type)
    print (data.cla_reg)
    print (data.num_class)

