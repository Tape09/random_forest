import numpy as np
import numpy.random as rnd
from scipy import stats
from Tree import *
from DataPool import *

class random_forest:  # {
    def __init__(self,data,data_type,y,y_type,n_classes,n_retry,number_of_trees, F, min_leaf_size=1, f_num='VR', f_cat='IG'):  # {
        self.trees=[]
        self.data=data
        self.y=y
        self.y_type=y_type
        for i in range(number_of_trees):
            #bootstrap/ divide data
            indices = np.random.permutation(data.shape[0])
            idx=int(data.shape[0]*2/3)
            training_idx, test_idx = indices[:idx], indices[idx:]
            training, test = data[training_idx,:], data[test_idx,:] 
            y_training= y[training_idx]
            #create tree
            new_tree=tree(training,data_type,y_training,y_type,n_classes,F,min_leaf_size,n_retry)
            #add tree to list
            tr=(new_tree,test_idx)
            self.trees.append(tr)

    # OOB
    def calculateOutOfBagError(self):
        error_list=[]
        for i,tup in enumerate(zip(self.data,self.y)):
            xi,yi=tup
            predictions=[]
            for tr,test_idx in self.trees:
                if i in test_idx:
                    pred=tr.predict(xi)
                    predictions.append(pred)
                    
            predictions=np.array(predictions)
            error_idxs=np.transpose(np.nonzero(predictions!=yi))
            error=len(error_idxs)/len(predictions)
            error_list.append(error)
            print(error)
        print('mean', np.mean(np.array(error_list))) 
            
            
            