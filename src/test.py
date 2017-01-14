import numpy as np
from scipy import stats
from RandomForest import *
from DataPool import *
import numpy.random as rnd;
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt


rnd.seed(9989)

# TODO: test mixed features, test regession


data = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[1,3],[1,4]])
#data = np.array([[0,0],[0,0]])
y = np.array([0,0,0,1,1,0,0,1,1])
#y = np.array([0,1])
data_type = np.ones(2)
y_type = 1
n_classes = np.array([2,7])
n_retry=100

y0idx = y==0
y0data = data[y0idx,:]
y1idx = y==1
y1data = data[y1idx,:]

dp=DataPool('glass.txt')
data=dp.data
data_type=dp.attribute_type
y=dp.class_v
y_type=dp.cla_reg
n_classes=np.zeros(data_type)
forest=random_forest(data,data_type,y,y_type,n_classes,n_retry,number_of_trees=50,F=1,min_leaf_size=1,f_num = "IG", f_cat = "IG")
print(forest.calculateOutOfBagError())


# I'm doing this through the terminal (ssh) so I can't plot. try plotting the strength against the number of trees (np.arange(0,50,1)). The plot should increase a lot at the start but converge towards teh end.
str,corr = forest.calculateStrengthAndCorrelation()




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

























