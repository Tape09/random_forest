import numpy as np
import numpy.random as rnd
from scipy import stats
import os
from DataPool import *

class node:  #{
    # Node class
    # left and right point to children
    # is_num decides if the split is on a numerical variable
    # split feature = index of feature that this node splits on
    # split_value = the split threshold for feature
    # value = if this is a leaf, the value that this node returns
    def __init__(self, is_num):
        self.left = None;
        self.right = None;
        self.is_num = is_num;
        self.split_feature = None;
        self.split_value = None;
        self.value = None;

    def predict(self, data_point,data_type):  # {
        if (self.is_leaf()):  # {
            return self.value;
        else:
            if (self.compare(data_point,data_type[self.split_feature])):
                return self.right.predict(data_point,data_type);
            else:
                return self.left.predict(data_point,data_type);
        # }

    # }

    def is_leaf(self):
        return (self.left == None and self.right == None)

    def compare(self, data_point, is_num):
        # helper function to do a split, based on is_num
        if (is_num==0):
            return data_point[self.split_feature] > self.split_value;
        else:
            return data_point[self.split_feature] == self.split_value;
#}


class dbg_node:
    def __init__(self,idxs):
        self.right = None;
        self.left = None;
        self.idxs = idxs;
        self.invalid = False;
        
        
    def is_leaf(self):
        return self.right == None and self.left == None


class lin_node:  #{
    # Node class
    # left and right point to children
    # is_num decides if the split is on a numerical variable
    # split feature = index of feature that this node splits on
    # split_value = the split threshold for feature
    # value = if this is a leaf, the value that this node returns
    def __init__(self):
        self.left = None;
        self.right = None;
        self.is_num = 0;
        self.split_features = None;
        self.split_coefs = None;
        self.split_value = None;
        self.value = None;
        self.split_subsets = None
        self.data_type = None

    def predict(self, data_point):  # {
        if (self.is_leaf()):  # {
            return self.value;
        else:
            if (self.compare(data_point)):
                return self.right.predict(data_point);
            else:
                return self.left.predict(data_point);
        # }

    # }

    def is_leaf(self):
        return (self.left == None and self.right == None)

    def compare(self, data_point):
        new_data = np.copy(data_point);
             
        #~ print(new_data)
        #~ print(self.split_features)
        #~ print(self.split_subsets)
        
        for j,f in enumerate(self.split_features):
            if(self.data_type[f] == 1):                
                if(new_data[f] in self.split_subsets[j]):
                    #~ print("asdf")
                    new_data[f] = 1;
                else:
                    new_data[f] = 0;
                    
        #~ print(new_data)
        #~ print()
        
        new_feature = new_data[self.split_features]
        new_feature = np.sum(self.split_coefs*new_feature)  
        
        #~ print(new_feature)
        
        return new_feature > self.split_value;

#}