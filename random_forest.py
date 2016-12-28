
import numpy as np
from scipy import stats


# 0= NUMERICAL ;;; 1= CATEGORICAL


def IG(data_left,y_left,data_right,y_right):	
	
def GINI(data_left,y_left,data_right,y_right):
	
def VR(data_left,y_left,data_right,y_right):

def import_datasets(folder):
	#return d x n x m feature data, d x m variable/feature type, d x n output, d x n output type d x m how many classes



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
		
		split_feature, split_number, data_left_idx, data_right_idx = find_split(data,data_type,y,n_classes);
		
		root = node(data_type[split_feature]);
		root.split_value = split_number;
		root.split_feature = split_feature;
				
		
		
		
	def grow_tree(self, root, data_left_idx, data_right_idx):
		uq_left = np.unique(self.y[data_left_idx])
		uq_right = np.unique(self.y[data_right_idx])
		
		if(len(data_left_idx) < self.min_leaf_size):
			#create leaf node
			node_left = node(self.y_type);
			if(y_type == 0):
				node_left.value = np.mean(self.y[data_left_idx]);
			else:
				node_left.value = stats.mode(self.y[data_left_idx]);
			root.left = node_left;
		else:
			if(len(uq_left) <= 1):
				node_left = node(self.y_type);
				node_left.value = uq_left;
			# UNDER CONSTRUCTION
			# if every y is the same - perfect split
				
			
		if(len(data_right_idx) < self.min_leaf_size):
			#create leaf node
			node_right = node(self.y_type);
			if(y_type == 0):
				node_right.value = np.mean(self.y[data_right_idx]);
			else:
				node_right.value = stats.mode(self.y[data_right_idx]);
			root.right = node_right;	
		
		
		
		
			
		
		
		
	#split_feature, split_number, data_left_idx1, data_right_idx1 = find_split(data[data_left_idx],data_type,y[data_left_idx],n_classes);
	
	
	def find_split(self, data, data_type,y,n_classes):
		best_feature = -1;
		best_split_number = -1;
		best_value = -999999;

		
		
		for feature in range(data.shape[1]): #{			
			if(data_type[feature] == 0): #{
				order = np.argsort(data[:,feature]);
				sorted_data = data[order,:];
				sorted_y = y[order,:]
				for split in range(data.shape[0]-1): #{						
					split_number = (sorted_data[split,feature] + sorted_data[split+1,feature])/2
					data_left = sorted_data[:(split+1),:]
					data_right = sorted_data[(split+1):,:]
					y_left = sorted_y[:(split+1),:]
					y_right = sorted_y[(split+1):,:]

					
					value = self.f_num(data_left,y_left,data_right,y_right);
					
					if(value > best_value):
						best_feature = feature;
						best_split_number = split_number;
						best_value = value;
						data_idx_left = order[:(split+1)];
						data_idx_right = order[(split+1):];
						

				#}
			#}	
			else: #{
				for split in range(n_classes[feature]): #{	
					idx_left = data[:,feature] != split
					idx_right = data[:,feature] == split
					data_left = data[idx_left,feature]
					data_right = data[idx_right,feature]
					y_left = y[idx_left]
					y_right = y[idx_right]
					
					value = self.f_cat(data_left,y_left,data_right,y_right);
					
					if(value > best_value):
						best_feature = feature;
						best_split_number = split;
						best_value = value;
						data_idx_left = idx_left;
						data_idx_right = idx_right;
						
					
				#}
				
			#}
		#}
		
		return best_feature,best_split_number,data_idx_left, data_idx_right
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















