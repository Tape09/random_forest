import numpy as np
import numpy.random as rnd
from scipy import stats
from node import *
from DataPool import *

import os

# Class for one tree
# The constructor takes in:
#
# data = feature matrix, first dimension is samples, second dimension is features
# data_type = vector over features; 0 if numeric, 1 if categorical
# y = class/regression label
# y_type = 0 if reg, 1 if cat
# n_classes = vector over features, number of possible classes for each feature. Only important for categorical features. Maybe can be combined with data_type
# F = number of features to be selected at each node
# min_leaf_size = minimum number of samples to make a split
# f_fum = the metric to be used for numerical
# f_cat = the metric to be used for categorical
class tree:  # { #UNDER CONSTRUCTION
    def __init__(self, data, data_type, y, y_type, n_classes, F = 1, min_leaf_size = 1 ,n_retry = 1,f_num = "VR", f_cat = "IG"):
        self.data = data;
        self.data_type = data_type;
        self.y = y;
        self.y_type = y_type;
        self.n_classes = n_classes;
        self.min_leaf_size = min_leaf_size;
        self.F = F;
        self.options = {"VR" : self.VR, "IG" : self.IG, "GINI" : self.GINI, "TEST" : self.TEST}
        self.f_num = self.options[f_num];
        self.f_cat = self.options[f_cat];
        self.n_retry = n_retry;

        split_feature, split_number, data_left_idx, data_right_idx = self.find_split(np.array(list(range(len(data)))));

        self.root = node(data_type[split_feature]);
        self.root.split_value = split_number;
        self.root.split_feature = split_feature;
        self.grow_tree(self.root, data_left_idx, data_right_idx);

    # ~ init();



    def init(self ,f_num = "VR", f_cat = "IG"):
        self.f_num = self.options[f_num];
        self.f_cat = self.options[f_cat];


    def predict(self, data_point):
        return self.root.predict(data_point, self.data_type);

    # TODO: VISUALIZATION??
    def visualize(self, bounds):
        # return a list of pairs of points
        # each pair defines a seperation line
        # bounds are the bounds for plotting, should be same as plt.axis(bounds)
        return 0;


    def grow_tree(self, root, data_left_idx, data_right_idx):  # {
        # recursive function for growing the tree
        # if all samples have same y - leaf node
        # if number of samples less than min_leaf_size - leaf node (should be called max leaf size)
        # else find best split and create child nodes
        # recurse on the child nodes
        uq_left = np.unique(self.y[data_left_idx])
        uq_right = np.unique(self.y[data_right_idx])
        left_leaf = False;
        right_leaf = False;
        #~ print(data_left_idx)
        if(len(data_left_idx) <= self.min_leaf_size):  # are there less than min_leaf_size samples?
            # create leaf node
            node_left = node(self.y_type);
            if(self. y_type == 0):
                node_left.value = np.mean(self.y[data_left_idx]);
            else:
                node_left.value = stats.mode(self.y[data_left_idx])[0][0];
            root.left = node_left;
            left_leaf = True;

        else:
            if(len( uq_left) <= 1):  # is there only one unique y left? Means all are same class.
                node_left = node(self.y_type);
                node_left.value = uq_left[0];
                root.left = node_left;
                left_leaf = True;



        if(len(data_right_idx) <= self.min_leaf_size)  : # are there less than min_leaf_size samples?
            # create leaf node
            node_right = node(self.y_type);
            if(self.y_type == 0):
                node_right.value = np.mean(self.y[data_right_idx]);
            else:
                node_right.value = stats.mode(self.y[data_right_idx])[0][0];
            root.right = node_right;
            right_leaf = True;

        else:
            if(len(uq_right) <= 1):   # is there only one unique y left? Means all are same class.
                node_right = node(self.y_type);
                node_right.value = uq_right[0];
                root.right = node_right;
                right_leaf = True;


        if(not right_leaf):  # {                #if right is not a leaf
            split_feature, split_number, data_left_idx1, data_right_idx1 = self.find_split(data_right_idx);
            
            # CHECK IF INVALID SPLIT (-1 RETURN)
            # MAKE LEAF NODE IF INVALID
            
            if(split_feature == -1):
                node_right = node(self.y_type);
                if(self.y_type == 0):
                    node_right.value = np.mean(self.y[data_right_idx]);
                else:
                    node_right.value = stats.mode(self.y[data_right_idx])[0][0];
                root.right = node_right;
                
            else:            
                node_right = node(self.data_type[split_feature]);
                node_right.split_feature = split_feature;
                node_right.split_value = split_number;

                root.right = node_right;
                self.grow_tree(root.right, data_left_idx1,data_right_idx1);
        # }

        if(not  left_leaf):  # {                     # if left is nto a leaf
            split_feature, split_number, data_left_idx1, data_right_idx1 = self.find_split(data_left_idx);
            
            # CHECK IF INVALID SPLIT (-1 RETURN)
            # MAKE LEAF NODE IF INVALID
            
            
            if(split_feature == -1):
                node_left = node(self.y_type);
                if(self.y_type == 0):
                    node_left.value = np.mean(self.y[data_left_idx]);
                else:
                    node_left.value = stats.mode(self.y[data_left_idx])[0][0];
                root.left = node_left;
                
            else:           
                node_left = node(self.data_type[split_feature]);
                node_left.split_feature = split_feature;
                node_left.split_value = split_number;

                root.left = node_left;
                self.grow_tree(root.left, data_left_idx1,data_right_idx1);
            # }

    # }



    def find_split(self,  data_idxs):  # {
        # Iterates over features and splits
        # Finds the best combination, according to the metric.
        # features are random based on F
        # splits are the feature classes for categorical
        # splits are between each feature value for numeric
        # returns best feature, best split, and the indexes of the data that are split to the left/right
        best_feature = -1;
        best_split_number = -1;
        best_value = -999999999;
        best_data_left_idx = -1;
        best_data_right_idx = -1;


        feature_idxs = np.array(list(range(self.data.shape[1])));
        for i in range(self.n_retry):
            random_features = np.array(rnd.choice( feature_idxs, self.F,replace = False));
            #~ print(random_features)

            for feature in random_features:  # {			 # loop over features
                if(self.data_type[feature]  == 0):  # {    # if numeric feature
                    order = np.argsort(self.data[ data_idxs,feature]);
                    sorted_data_idxs = data_idxs[order];
                    
                    uq_values, uq_idx = np.unique(self.data[ sorted_data_idxs, feature],return_index = True);
                        
                    #~ print(uq_values)
                    #~ print(uq_idx)
                    #~ print(sorted_data_idxs)
                    for split in range(len (uq_idx)-1):  # {       #loop over splits
                        #print(uq_values)
                        split_number = (uq_values[split] + uq_values [ split+1])/2
                        data_left_idx = data_idxs[self.data[ data_idxs,feature] < split_number]
                        data_right_idx = data_idxs[self.data[ data_idxs,feature] >= split_number]

                        value = self.f_num( data_left_idx,data_right_idx);

                        if(value > best_value):
                            best_feature = feature;
                            best_split_number = split_number;
                            best_value = value;
                            best_data_left_idx = data_left_idx;
                            best_data_right_idx = data_right_idx;


                    # }
                # }	  
                else:  # {             # if cat feature
                    
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
                    # TODO: CHECK IF ALL FEATURE VALUES ARE THE SAME, SOLVE IT SAME WAY AS FOR NUM #
                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
                    
                    for split in range(self.n_classes[feature]):  # {	         #loop over splits
                        idx_left = data_idxs[self.data[data_idxs, feature] != split]
                        idx_right = data_idxs[self.data[data_idxs, feature] == split]

                        value = self.f_cat(idx_left, idx_right);

                        if(value > best_value):
                            best_feature = feature;
                            best_split_number = split;
                            best_value = value;
                            best_data_left_idx = idx_left;
                            best_data_right_idx = idx_right;


                    # }
                # }
            # }
            if not isinstance(best_data_left_idx,int) and not isinstance(best_data_right_idx,int):
                break;


        return best_feature, best_split_number, best_data_left_idx, best_data_right_idx

    # }



    # input is indexes over data
    # these are member functions, so you can use self.data
    # should return a float
    def IG(self, data_left_idx, data_right_idx):
        return 0
    def GINI(self, data_left_idx, data_right_idx):
        return 0

    def VR(self, data_left_idx, data_right_idx):
        return 0

    def TEST(self, data_left_idx, data_right_idx):
        #print(data_left_idx,data_right_idx,self.y)
        return -(np.var(self.y[data_left_idx]) + np.var(self.y[data_right_idx]))


# }