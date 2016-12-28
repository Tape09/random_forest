
import numpy as np
from scipy import stats


# 0= NUMERICAL ;;; 1= CATEGORICAL



def import_datasets(name_data, data):
    if name_data == 'glass.txt':
        dataset = data[:, 1:-1]
        [num_samples, num_features] = dataset.shape
        class_v = data[:,-1]
        cat = 1
        return dataset, num_samples, num_features, class_v, cat

#   # d = dataset
# 	# n = n_samples
# 	# m = n_features
#
# 	# categorical variables should be indexes (0,1,2,3 not "R" etc.) remember mapping
# 	# type should be 0=numeric, 1=categorical
# 	# output type 0=regression, 1=categorical





def readdata(datapath):
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
    with open(datapath,"r") as f:
        mylist = f.read().splitlines()
        for line in mylist:
            currentline = line.split(",")
            data_l.append(currentline)
    data = np.array(data_l)
    return data

def get_paths(data_name):
    '''
        Get the project path and training data path
        :param data_name: the name of the test data
        :return: training data path
        '''
    src_path = os.getcwd()
    project_path = os.path.dirname(src_path)
    data_path = os.path.join(project_path, 'data')
    test_data_dir = os.path.join(data_path, data_name)
    return test_data_dir

def demo():
    name_data = 'glass.txt'
    datapath = get_paths(name_data)  # give the name of the data file, return the accessible path
    data = readdata(datapath)
    dataset, num_samples, num_features, class_v, cat = import_datasets(name_data, data)
    
    print dataset, num_samples, num_features, class_v, cat




class random_forest: #{
	def __init__(number_of_trees, F, min_leaf_size = 1, f_num = VR, f_cat = IG): #{
		
	def add_data(#n x m feature data, m variable/feature type, n output, training_split = 0.1)


	#}





#}


class tree: #{ #UNDER CONSTRUCTION
	def __init__(data, data_type, y, y_type, n_classes, min_leaf_size = 1, f_num = VR, f_cat = IG):
		self.data = data;
		self.data_type = data_type;
		self.y = y;
		self.y_type = y_type;
		self.n_classes = n_classes;
		self.min_leaf_size = min_leaf_size;
		self.f_num = f_num;
		self.f_cat = f_cat;
		
		split_feature, split_number, data_left_idx, data_right_idx = find_split(list(range(len(data))));
		
		self.root = node(data_type[split_feature]);
		self.root.split_value = split_number;
		self.root.split_feature = split_feature;
				
		
		
		
	def grow_tree(self, root, data_left_idx, data_right_idx):
		uq_left = np.unique(self.y[data_left_idx])
		uq_right = np.unique(self.y[data_right_idx])
		left_leaf = False;
		right_leaf = False;
		
		if(len(data_left_idx) < self.min_leaf_size):
			#create leaf node
			node_left = node(self.y_type);
			if(y_type == 0):
				node_left.value = np.mean(self.y[data_left_idx]);
			else:
				node_left.value = stats.mode(self.y[data_left_idx]);
			root.left = node_left;
			left_leaf = True;
			
		else:
			if(len(uq_left) <= 1):
				node_left = node(self.y_type);
				node_left.value = uq_left[0];
				root.left = node_left;
				left_leaf = True;
				
				
			
		if(len(data_right_idx) < self.min_leaf_size):
			#create leaf node
			node_right = node(self.y_type);
			if(y_type == 0):
				node_right.value = np.mean(self.y[data_right_idx]);
			else:
				node_right.value = stats.mode(self.y[data_right_idx]);
			root.right = node_right;
			right_leaf = True;
			
		else:
			if(len(uq_right) <= 1):
				node_right = node(self.y_type);
				node_right.value = uq_right[0];
				root.right = node_right;
				right_leaf = True;
		
		
		if(not right_leaf) #{
			split_feature, split_number, data_left_idx1, data_right_idx1 = find_split(data_right_idx);
			node_right = node(self.n_classes[split_feature]);
			node_right.split_feature = split_feature;
			node_right.split_value = split_number;
			
			root.right = node_right;
			grow_tree(root.right,data_left_idx1,data_right_idx1);		
		
		
		#}
		
		if(not left_leaf) #{
		
		
		
		
		#}		
		
		
		
	#split_feature, split_number, data_left_idx1, data_right_idx1 = find_split(data[data_left_idx],data_type,y[data_left_idx],n_classes);
	
	
	def find_split(self, data_idxs): #{
		best_feature = -1;
		best_split_number = -1;
		best_value = -999999;
		best_data_left_idx = -1;
		best_data_right_idx = -1;
		
		
		for feature in range(data.shape[1]): #{			
			if(data_type[feature] == 0): #{
				order = np.argsort(data[data_idxs,feature]);
				sorted_data_idxs = data_idxs[order];
				sorted_y = y[sorted_data_idxs,:]
				for split in range(len(sorted_data_idxs)-1): #{						
					split_number = (self.data[sorted_data_idxs[split],feature] + self.data[sorted_data_idxs[split+1],feature])/2
					data_left_idx = sorted_data_idxs[:(split+1),:]
					data_right_idx = sorted_data_idxs[(split+1):,:]
					
					value = self.f_num(data_left_idx,data_right_idx);
					
					if(value > best_value):
						best_feature = feature;
						best_split_number = split_number;
						best_value = value;
						best_data_left_idx = data_left_idx;
						best_data_right_idx = data_right_idx;
						

				#}
			#}	
			else: #{
				for split in range(n_classes[feature]): #{	
					idx_left = data_idxs[self.data[data_idxs,feature] != split]
					idx_right = data_idxs[self.data[data_idxs,feature] == split]
					
					value = self.f_cat(data_left_idx,data_right_idx);
					
					if(value > best_value):
						best_feature = feature;
						best_split_number = split;
						best_value = value;
						best_data_left_idx = idx_left;
						best_data_right_idx = idx_right;
						
					
				#}
				
			#}
		#}
		
		return best_feature,best_split_number,best_data_left_idx, best_data_right_idx
	#}
	
	
	
	# input is indexes over data
	# these are member functions, so you can use self.data
	# should return a float
	def IG(data_left_idx,data_right_idx):
	
	def GINI(data_left_idx,data_right_idx):
		
	def VR(data_left_idx,data_right_idx):
	
	
#}
	
	
	
	
	
class node: #{ #UNDER CONSTRUCTION
	def __init__(is_num):
		self.left = None;
		self.right = None;
		self.is_num = is_num;
		self.split_feature = None;
		self.split_value = None;
		self.value = None;

	def is_leaf():
		return (self.left == None and self.right == None)


#}















