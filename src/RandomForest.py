import numpy as np
import numpy.random as rnd
from scipy import stats
from Tree import *
from lin_tree import *
from DataPool import *

class random_forest:  # {
    def __init__(self,data,data_type,y,y_type,n_classes,n_retry,number_of_trees, F, min_leaf_size=1, rc = False, L = 3, f_num='VR', f_cat='IG'):  # {
        self.trees=[]
        self.bags=[]
        self.n_trees = number_of_trees;
        self.data=data
        self.y=y
        self.y_type=y_type        
        self. rc = rc;
        self.L = L;

        self.f_num = f_num;
        self.f_cat = f_cat;
        for i in range(number_of_trees):
            #bootstrap/ divide data
            bag_indices = rnd.choice(data.shape[0],data.shape[0])
            training = data[bag_indices,:]
            y_training= y[bag_indices]
            #create tree
            if(rc):
                new_tree=lin_tree(training,data_type,y_training,y_type,n_classes,n_retry=n_retry,F=F,L=L,min_leaf_size=min_leaf_size,f_num = self.f_num, f_cat = self.f_cat)
                
                #~ new_tree=lin_tree(data_train,data_type,y_train,y_type,n_classes,n_retry=n_retry,F=2,L=3, min_leaf_size=1,f_num = "IG", f_cat = "IG")
            else:
                new_tree=tree(training,data_type,y_training,y_type,n_classes,F,min_leaf_size,n_retry,f_num = self.f_num, f_cat = self.f_cat)
            
            
            #~ preds = []
            #~ for i in range(training.shape[0]):
                #~ preds.append(new_tree.predict(data[i]));

            #~ preds = np.array(preds);
            #~ correct = preds == y 
            #~ print(float(np.sum(correct))/training.shape[0])
            
            
            #add tree to list
            self.trees.append(new_tree)
            self.bags.append(bag_indices)

    # OOB
    def calculateOutOfBagError(self):
        error_list=[]
        for i,tup in enumerate(zip(self.data,self.y)):
            xi,yi=tup
            predictions=[]
            for j in range(len(self.trees)):
                if i not in self.bags[j]:
                    pred=self.trees[j].predict(xi)
                    predictions.append(pred)
            
            if(self.y_type == 1):
                error_list.append(0.0 if stats.mode(predictions)[0][0] == yi else 1.0);
            else:
                error_list.append((np.mean(predictions)-yi)**2);                
            
        # print('mean', np.mean(np.array(error_list))) 
        return(np.mean(np.array(error_list)))
        
    def calculateOutOfBagTreeError(self):
        error_list=[]
        
        for j in range(len(self.trees)):   
            wrong = 0.0;
            total = 0.0;
            for i,tup in enumerate(zip(self.data,self.y)):
                xi,yi=tup               
                if i not in self.bags[j]:
                    pred=self.trees[j].predict(xi)
                    if pred != yi:
                        wrong += 1
                        
                    total += 1
            error_list.append(float(wrong)/total)
    
        return(np.mean(np.array(error_list)))
    

    def calculateTestError(self,data,y):
        error_list=[]
        
        for i,tup in enumerate(zip(data,y)):
            xi,yi=tup
            predictions=[]
            for j in range(len(self.trees)):                
                pred=self.trees[j].predict(xi)
                predictions.append(pred)
            
            if(self.y_type == 1):
                error_list.append(0.0 if stats.mode(predictions)[0][0] == yi else 1.0);
            else:
                error_list.append((np.mean(predictions)-yi)**2);                
            
        # print('mean', np.mean(np.array(error_list))) 
        return(np.mean(np.array(error_list)))
            

    # Calculated the strength and correlation as described in the paper by breiman.
    # returns two arrays. correlation and strength
    # they are arrays because we calculate str and corr after adding each tree. If you want the total strength and correlation of the forest jsut take the last value of the arrays. [-1]
    def calculateStrengthAndCorrelation(self,last=True):
        
        if(last):
            offset = self.n_trees - 1
        else:
            offset = 0;
        
        strength = np.zeros(self.n_trees - offset)
        correlation = np.zeros(self.n_trees - offset)
        
        y_classes = np.unique(self.y)
        n_y_classes = len(y_classes)
        
        
            
        for m in range(self.n_trees-offset): #{
            K = m + offset            
            
            Q = np.zeros((self.data.shape[0],n_y_classes))
            temp_str = [];
            jhats = [];
            
            for d in range(self.data.shape[0]): #{
                maxQj = -9999999;
                jhat = -1;
                y_idx = np.where(y_classes==self.y[d])[0][0];
                
                sumsj = np.zeros(n_y_classes);
                sumsjall = np.zeros(n_y_classes);
                
                for j in range(n_y_classes): #{
                    sum_j = 0.0;
                    sum_all = 0.0;
                    for k in range(K+1): #{
                        #~ print(k)
                        if(d not in self.bags[k]):
                            if(self.trees[k].predict(self.data[d])==y_classes[j]):
                                sum_j+=1;
                            sum_all += 1;
                    #}
                    
                    sumsj[j] = sum_j;
                    sumsjall[j] = sum_all;
                    
                    if(sum_all == 0):
                        Q[d,j] = 0;
                    else:
                        Q[d,j] = float(sum_j)/sum_all
                    
                    if(y_idx != j):
                        if(maxQj < Q[d,j]):
                            maxQj = Q[d,j]
                            jhat = j;
                #}
                jhats.append(jhat);
                temp_str.append(Q[d,y_idx] - maxQj);               
            #}
            strength[m] = np.mean(temp_str);
            varmr = np.mean(np.square(temp_str))-strength[m]**2
            
            p1 = [];
            p2 = [];
            for k in range(K+1): #{
                p1k = 0.0;
                p2k = 0.0;
                for d in range(self.data.shape[0]): #{
                    if(self.trees[k].predict(self.data[d])==self.y[d]):
                        p1k += 1.0;
                    if(self.trees[k].predict(self.data[d])==y_classes[jhats[d]]):
                        p2k += 1.0;
                #}
                p1.append(p1k/self.data.shape[0])                
                p2.append(p2k/self.data.shape[0])                
            #}
            
            p1 = np.array(p1)
            p2 = np.array(p2)
            
            stds = np.sqrt(p1 + p2 + np.square(p1-p2))            
            std = np.mean(stds);
            
            correlation[m] = varmr/(std**2)            
        #}
        
        return strength,correlation;
            