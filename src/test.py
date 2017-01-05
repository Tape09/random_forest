import numpy as np
from scipy import stats
from RandomForest import *
from DataPool import *
from Tree import tree
from testerror import *
import matplotlib.pyplot as plt

# TODO: test mixed features, test regession

data = np.array([[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[1,3],[1,4]])
y = np.array([0,0,0,1,1,0,0,1,1])
data_type = np.zeros(9)
y_type = 1
n_classes = np.array([2,7])

y0idx = y==0
y0data = data[y0idx,:]
y1idx = y==1
y1data = data[y1idx,:]



#~ plt.figure()
#~ plt.plot(y0data[:,0],y0data[:,1],'go')
#~ plt.plot(y1data[:,0],y1data[:,1],'rx')
#~ plt.axis([-0.5,1.5,-0.5,6.5])
#~ plt.show()


t = tree(data, data_type, y, y_type, n_classes, f_num = "TEST", f_cat = "TEST");
#~ t.init("TEST","TEST")
root = t.root
left = root.left
right = root.right


def run_testerror():
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

run_testerror()








