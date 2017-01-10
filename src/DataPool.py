#! /usr/bin/env python
import numpy as np
import os

class DataPool:
    """
    The collection of all the data
    Take use of the methods to retrieve the data we need
    """
    def __init__(self, name_data):
        self.name_data = name_data
        self.data = self.__readdata(name_data+'_features.csv')  # only the data
        [self.num_samples, self.num_features] = self.data.shape
        self.class_v = self.__readdata(name_data+'_y.csv').flatten()

        '''
         numerical (real?) = 0
         categorical (integer?) = 1
        '''
        attribute_type_str = self.__readdata(name_data+'_attribute_type.csv').flatten()
        self.attribute_type = [int(x) for x in attribute_type_str]

        '''
        classification tree: cla_reg = 1
        regression tree : cla_reg = 0
        '''
        self.cla_reg = int(self.__readdata(name_data+'_clas_reg_tree.csv').flatten()[0])

        self.num_class_temp = []
        self.num_class = []

        for i in range(0, self.num_features):
            if self.attribute_type[i] == 1:
                for j in range(0, self.num_samples):
                    self.num_class_temp.append(self.data[j][i])
                self.num_class.append( len(list(set(self.num_class_temp))))
                self.num_class_temp = []
            else:
                self.num_class.append(-1)




    # data_type, y, y_type, n_classes, min_leaf_size = 1, n_retry = 1,
    def __get_paths(self):
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
        return data_path

    def __readdata(self, fileName):
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
        data_file = os.path.join(self.__get_paths(), fileName)  # now we are in the folder containing the training data
        data_l = []
        with open(data_file, "r") as f:
            mylist = f.read().splitlines()
            for line in mylist:
                currentline = line.split(",")
                data_l.append(currentline)
        data = np.array(data_l)
        return data


def demo():
    data = DataPool('glass')
    #data = DataPool('diabetes')
    #data = DataPool('sonar')
    #data = DataPool('ionosphere')
    #data = DataPool('vehicle')
    #data = DataPool('soybean')
    #data = DataPool('german')
    #data = DataPool('image_statlog')
    #data = DataPool('ecoli')
    #data = DataPool('votes')
    #data = DataPool('liver')
    #data = DataPool('letter-recognition')
    #data = DataPool('sat.images.training')
    #data = DataPool('sat.images.testing')
    #data = DataPool('waveform')
    print (data.data)
    print (data.class_v)
    print (data.attribute_type)
    print (data.cla_reg)
    print (data.num_class)
demo()
