import numpy as np
from scipy import stats
from RandomForest import *
from DataPool import *
import numpy.random as rnd;
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import sys
import time

#import matplotlib.pyplot as plt


#~ rnd.seed(9989)

# TODO: test mixed features, test regession


#~ data = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[1,3],[1,4]])
#~ #data = np.array([[0,0],[0,0]])
#~ y = np.array([0,0,0,1,1,0,0,1,1])
#~ #y = np.array([0,1])
#~ data_type = np.ones(2)
#~ y_type = 1
#~ n_classes = np.array([2,7])


#~ y0idx = y==0
#~ y0data = data[y0idx,:]
#~ y1idx = y==1
#~ y1data = data[y1idx,:]



n_retry = 20

dataset  = sys.argv[1]


print("dataset:",dataset)

dp=DataPool(dataset)
data=dp.data
data_type=dp.attribute_type
y=dp.class_v
y_type=dp.cla_reg
n_classes=dp.num_class



n_iterations = 25

final_oob = np.zeros(50)
final_test = np.zeros(50)
final_str = np.zeros(50)
final_cor = np.zeros(50)


for F in range(1,15):
    print("F=",F)
    oob_errs = []
    #~ oob_tree_errs = []
    test_errs = []
    strs = []
    cors = []
    for i in range(n_iterations):
        n = data.shape[0]
        n_train = int(n*0.9)
        n_test = n - n_train;

        test_idxs = rnd.choice(np.arange(n),n_test,replace=False)
        
        train_mask = np.ones(n,dtype=bool)
        train_mask[test_idxs] = False
        
        data_train = data[train_mask,:]
        y_train = y[train_mask]
        data_test = data[~train_mask,:]
        y_test = y[~train_mask]
        
        #~ forest=random_forest(data_train,data_type,y_train,y_type,n_classes,n_retry,number_of_trees=100,F=1,min_leaf_size=1,f_num = "IG", f_cat = "IG")
        forest=random_forest(data_train,data_type,y_train,y_type,n_classes,n_retry,number_of_trees=100,F=F,rc=False,min_leaf_size=1,f_num = "IG", f_cat = "IG")    
        
        oob_errs.append(forest.calculateOutOfBagError())
        #~ oob_tree_errs.append(forest.calculateOutOfBagTreeError())
        test_errs.append(forest.calculateTestError(data_test,y_test))
        s,c = forest.calculateStrengthAndCorrelation()
        
        strs.append(s[0])
        cors.append(c[0])
        
        sys.stdout.flush()
        sys.stdout.write("\r" + str(100*float(i+1)/n_iterations) + "%")
        
        
    print()

    oob_errs = np.array(oob_errs)
    #~ oob_tree_errs = np.array(oob_tree_errs)
    test_errs = np.array(test_errs)
    strs = np.array(strs)
    cors = np.array(cors)
    
    final_oob[F] = np.mean(oob_errs)
    final_test[F] = np.mean(test_errs)
    final_str[F] = np.mean(strs)
    final_cor[F] = np.mean(cors)
    
    
plt.figure()
plt.plot(np.arange(50)+1,final_oob,color = "black")
plt.plot(np.arange(50)+1,final_test, color = "red")
plt.xlabel("F")
plt.ylabel("Error")
plt.title("OOB and test error vs F")
plt.savefig("ooberr1.png")



plt.figure()
plt.plot(np.arange(50)+1,final_str,color = "black")
plt.plot(np.arange(50)+1,final_cor, color = "red")
plt.xlabel("F")
plt.ylabel("Error")
plt.title("OOB and test error vs F")
plt.savefig("strcor1.png")



#~ print("dataset:",dataset)
#~ print("Out-Of-Bag Error:",np.mean(oob_errs))
#~ print("Out-Of-Bag Tree Error:",np.mean(oob_tree_errs))
#~ print("Test Error:",np.mean(test_errs))


# I'm doing this through the terminal (ssh) so I can't plot. try plotting the strength against the number of trees (np.arange(0,50,1)). The plot should increase a lot at the start but converge towards teh end.
#~ str,corr = forest.calculateStrengthAndCorrelation()




#~ plt.figure()
#~ plt.plot(y0data[:,0],y0data[:,1],'go')
#~ plt.plot(y1data[:,0],y1data[:,1],'rx')
#~ plt.axis([-0.5,1.5,-0.5,6.5])
#~ plt.show()


# n = data.shape[0];
# m = 0;
# #~ n = 155;
# t = tree(data[m:n], data_type, y[m:n], y_type, n_classes, n_retry = n_retry, f_num = "VR", f_cat = "VR");


# preds = []
# for i in range(data[m:n].shape[0]):
    # preds.append(t.predict(data[i]));


# preds = np.array(preds);

# correct = preds == y[m:n]

# #~ print(correct)



# #~ t.init("TEST","TEST")
# root = t.root
# left = root.left
# right = root.right

# bad_node = t.root.left.right
# bad_dnode = t.debug_root.left.right


# rfc = RandomForestClassifier(50,criterion="entropy",warm_start=True, max_features=1,bootstrap=True,oob_score=True,random_state=9989)

# rfc.fit(data,y)

























