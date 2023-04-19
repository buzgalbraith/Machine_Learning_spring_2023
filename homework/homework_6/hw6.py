# %%
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
import graphviz

from IPython.display import Image

#%matplotlib inline

# %% [markdown]
# # Load Data

# %%
data_train = np.loadtxt('svm-train.txt')
data_test = np.loadtxt('svm-test.txt')
x_train, y_train = data_train[:, 0: 2], data_train[:, 2].reshape(-1, 1)
x_test, y_test = data_test[:, 0: 2], data_test[:, 2].reshape(-1, 1)

# %%
# Change target to 0-1 label
y_train_label = np.array(list(map(lambda x: 1 if x > 0 else 0, y_train))).reshape(-1, 1)

# %% [markdown]
# # Decision Tree Class

# %%
def compute_entropy(label_array):
    '''
    Calulate the entropy of given label list
    
    :param label_array: a numpy array of binary labels shape = (n, 1)
    :return entropy: entropy value
    '''
    proportion_observations = lambda label_array : np.array([( np.sum(label_array==class_name) )/label_array.shape[0] for class_name in np.unique(label_array)])
    proportions = proportion_observations(label_array)
    entropy = -np.sum(proportions * np.log(proportions))
    return entropy

def compute_gini(label_array):
    '''
    Calulate the gini index of label list
    
    :param label_array: a numpy array of labels shape = (n, 1)
    :return gini: gini index value
    '''
    proportion_observations = lambda label_array : np.array([( np.sum(label_array==class_name) )/label_array.shape[0] for class_name in np.unique(label_array)])
    proportions = proportion_observations(label_array)
    gini = np.sum(proportions * (1-proportions))
    return gini

# %%
class Decision_Tree(BaseEstimator):
     
    def __init__(self, split_loss_function, leaf_value_estimator,
                 depth=0, min_sample=5, max_depth=10):
        '''
        Initialize the decision tree classifier

        :param split_loss_function: method with args (X, y) returning loss
        :param leaf_value_estimator: method for estimating leaf value from array of ys
        :param depth: depth indicator, default value is 0, representing root node
        :param min_sample: an internal node can be splitted only if it contains points more than min_smaple
        :param max_depth: restriction of tree depth.
        '''
        self.split_loss_function = split_loss_function
        self.leaf_value_estimator = leaf_value_estimator
        self.depth = depth
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.is_leaf = False
        # print("max depth set as ", self.max_depth)
        # print("min samples set as ",self.min_sample)
    def fit(self, x, y):
        '''
        This should fit the tree classifier by setting the values self.is_leaf, 
        self.split_id (the index of the feature we want ot split on, if we're splitting),
        self.split_value (the corresponding value of that feature where the split is),
        and self.value, which is the prediction value if the tree is a leaf node.  If we are 
        splitting the node, we should also init self.left and self.right to be Decision_Tree
        objects corresponding to the left and right subtrees. These subtrees should be fit on
        the data that fall to the left and right,respectively, of self.split_value.
        This is a recurisive tree building procedure. 
        
        :param X: a numpy array of training data, shape = (n, m)
        :param y: a numpy array of labels, shape = (n, 1)

        :return self
        '''
        try:
            self.features=np.array(list(range(x.shape[1])))
        except:
            self.features=np.array(list(range(1)))
        if(self.depth>=self.max_depth or x.shape[0]<=self.min_sample):
            self.is_leaf=True
            self.value=self.leaf_value_estimator(y)
        else:
            self.find_best_feature_split(x, y)
            # print(self.split_value)
            feature_vector = x[:,self.split_id]
            left_x=x[feature_vector<self.split_value]
            left_y=y[feature_vector<self.split_value]
            right_x=x[feature_vector>=self.split_value]
            right_y=y[feature_vector>=self.split_value]
            if min(left_x.shape[0], right_x.shape[0])<self.min_sample:
                self.is_leaf=True
                self.value=self.leaf_value_estimator(y)
            else:
                self.left=Decision_Tree(split_loss_function=self.split_loss_function, leaf_value_estimator=self.leaf_value_estimator,depth=self.depth+1, min_sample=self.min_sample,
                                        max_depth=self.max_depth)
                self.right=Decision_Tree(split_loss_function=self.split_loss_function, leaf_value_estimator=self.leaf_value_estimator,depth=self.depth+1, min_sample=self.min_sample,
                                        max_depth=self.max_depth)
                self.left.fit(left_x, left_y)
                self.right.fit(right_x, right_y)
        return self

    def find_best_split(self, x_node, y_node, feature_id):
        '''
        For feature number feature_id, returns the optimal splitting point 
        for data X_node, y_node, and corresponding loss
        :param X: a numpy array of training data, shape = (n_node)
        :param y: a numpy array of labels, shape = (n_node, 1)
        '''
        potential_splits=x_node.shape[0]-1
        feature_vector=x_node[:,feature_id]
        zipped_sort=np.array(sorted(zip(feature_vector, y_node), key=lambda x: x[0]))
        losses=np.ones((x_node.shape[0],2))
        for i in range(1,x_node.shape[0]-1):
            left_split_labels=zipped_sort[zipped_sort[:,0]<zipped_sort[:,0][i]][:,1]
            right_split_labels=zipped_sort[zipped_sort[:,0]>=zipped_sort[:,0][i]][:,1]
            numerator=(left_split_labels.shape[0] * self.split_loss_function(left_split_labels))+(right_split_labels.shape[0] * self.split_loss_function(right_split_labels))
            node_impurity=numerator/(left_split_labels.shape[0] + right_split_labels.shape[0])
            losses[i,0]=node_impurity
            losses[i,1]=zipped_sort[:,0][i]
        cut_off=min(potential_splits//2,self.min_sample)
        losses_to_consider=losses[cut_off:x_node.shape[0]-cut_off,:]
        best_loss=min(losses_to_consider[:,0])
        split_value=losses_to_consider[np.argmin(losses_to_consider[:,0]),1]
        

        return split_value, best_loss

        
    def find_best_feature_split(self, x_node, y_node):
        '''
        Returns the optimal feature to split and best splitting point 
        for data X_node, y_node.
        :param X: a numpy array of training data, shape = (n_node, 1)
        :param y: a numpy array of labels, shape = (n_node, 1)
        '''
        feature_loss= np.zeros(self.features.shape[0])
        feature_split= np.zeros(self.features.shape[0])
        for i in range(self.features.shape[0]):
            optimal_split, optimal_split_loss = self.find_best_split( x_node, y_node, self.features[i])
            feature_loss[i]=optimal_split_loss
            feature_split[i]=optimal_split
        best_split_index = np.argmin(feature_loss)
        best_split=feature_split[best_split_index]
        best_feature=self.features[best_split_index]
        self.split_id = best_feature
        self.split_value = best_split
        return self.split_id, self.split_value


    def predict_instance(self, instance):
        '''
        Predict label by decision tree

        :param instance: a numpy array with new data, shape (1, m)

        :return whatever is returned by leaf_value_estimator for leaf containing instance
        '''
        if self.is_leaf:
            return self.value
        if instance[self.split_id] <= self.split_value:
            return self.left.predict_instance(instance)
        else:
            return self.right.predict_instance(instance)

# %% [markdown]
# # Decision Tree Classifier

# %%
def most_common_label(y):
    '''
    Find most common label
    '''
    label_cnt = Counter(y.reshape(len(y)))
    label = label_cnt.most_common(1)[0][0]
    return label

# %%
class Classification_Tree(BaseEstimator, ClassifierMixin):

    loss_function_dict = {
        'entropy': compute_entropy,
        'gini': compute_gini
    }

    def __init__(self, loss_function='entropy', min_sample=5, max_depth=10):
        '''
        :param loss_function(str): loss function for splitting internal node
        '''
        self.loss_function=loss_function
        self.max_depth=max_depth
        self.min_sample=min_sample
        self.tree = Decision_Tree(self.loss_function_dict[loss_function],
                                most_common_label,
                                0, min_sample, max_depth)

    def fit(self, X, y=None):
        self.tree.fit(X,y)
        return self

    def predict_instance(self, instance):
        value = self.tree.predict_instance(instance)
        return value

# %% [markdown]
# # Decision Tree Boundary

# %%
# Training classifiers with different depth
clf1 = Classification_Tree(max_depth=1, min_sample=2)
clf1.fit(x_train, y_train_label)

clf2 = Classification_Tree(max_depth=2, min_sample=2)
clf2.fit(x_train, y_train_label)

clf3 = Classification_Tree(max_depth=3, min_sample=2)
clf3.fit(x_train, y_train_label)

clf4 = Classification_Tree(max_depth=4, min_sample=2)
clf4.fit(x_train, y_train_label)

clf5 = Classification_Tree(max_depth=5, min_sample=2)
clf5.fit(x_train, y_train_label)

clf6 = Classification_Tree(max_depth=6, min_sample=2)
clf6.fit(x_train, y_train_label)

# Plotting decision regions
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1, 2]),
                        [clf1, clf2, clf3, clf4, clf5, clf6],
                        ['Depth = {}'.format(n) for n in range(1, 7)]):
    Z = np.array([clf.predict_instance(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(x_train[:, 0], x_train[:, 1], c=y_train_label[:,0], alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()

# %% [markdown]
# # Compare decision tree with tree model in sklearn

# %%
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=2)
clf.fit(x_train, y_train_label)
export_graphviz(clf, out_file='tree_classifier.dot')

# %%
# Plotting decision regions
x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = np.array([clf.predict(x[np.newaxis,:]) for x in np.c_[xx.ravel(), yy.ravel()]])
Z = Z.reshape(xx.shape)
plt.figure()
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(x_train[:, 0], x_train[:, 1], 
c=y_train_label[:,0], alpha=0.8)

# %%
# Visualize decision tree
#!dot -Tpng tree_classifier.dot -o tree_classifier.png
Image(filename='tree_classifier.png')

# %% [markdown]
# # Decision Tree Regressor

# %%
# Regression Tree Specific Code
def mean_absolute_deviation_around_median(y):
    '''
    Calulate the mean absolute deviation around the median of a given target list
    
    :param y: a numpy array of targets shape = (n, 1)
    :return mae
    '''
    y=np.sort(y)
    if(y.shape[0]%2==0):
        median=(y[y.shape[0]//2]+y[y.shape[0]//2-1])/2
    else:
        median=y[y.shape[0]//2]
    mae=np.mean(np.abs(y-median))
    return mae

# %%
class Regression_Tree():
    '''
    :attribute loss_function_dict: dictionary containing the loss functions used for splitting
    :attribute estimator_dict: dictionary containing the estimation functions used in leaf nodes
    '''

    loss_function_dict = {
        'mse': np.var,
        'mae': mean_absolute_deviation_around_median
    }

    estimator_dict = {
        'mean': np.mean,
        'median': np.median
    }
    
    def __init__(self, loss_function='mse', estimator='mean', min_sample=5, max_depth=10):
        '''
        Initialize Regression_Tree
        :param loss_function(str): loss function used for splitting internal nodes
        :param estimator(str): value estimator of internal node
        '''

        self.tree = Decision_Tree(self.loss_function_dict[loss_function],
                                  self.estimator_dict[estimator],
                                  0, min_sample, max_depth)

    def fit(self, X, y=None):
        self.tree.fit(X,y)
        return self

    def predict_instance(self, instance):
        value = self.tree.predict_instance(instance)
        return value

# %% [markdown]
# # Fit regression tree to one-dimensional regression data

# %%
data_krr_train = np.loadtxt('krr-train.txt')
data_krr_test = np.loadtxt('krr-test.txt')
x_krr_train, y_krr_train = data_krr_train[:,0].reshape(-1,1),data_krr_train[:,1].reshape(-1,1)
x_krr_test, y_krr_test = data_krr_test[:,0].reshape(-1,1),data_krr_test[:,1].reshape(-1,1)

# Training regression trees with different depth
clf1 = Regression_Tree(max_depth=1,  min_sample=3, loss_function='mae', estimator='mean')
clf1.fit(x_krr_train, y_krr_train)

clf2 = Regression_Tree(max_depth=2,  min_sample=3, loss_function='mae', estimator='mean')
clf2.fit(x_krr_train, y_krr_train)

clf3 = Regression_Tree(max_depth=3,  min_sample=3, loss_function='mae', estimator='mean')
clf3.fit(x_krr_train, y_krr_train)

clf4 = Regression_Tree(max_depth=4,  min_sample=3, loss_function='mae', estimator='mean')
clf4.fit(x_krr_train, y_krr_train)

clf5 = Regression_Tree(max_depth=5,  min_sample=3, loss_function='mae', estimator='mean')
clf5.fit(x_krr_train, y_krr_train)

clf6 = Regression_Tree(max_depth=10,  min_sample=3, loss_function='mae', estimator='mean')
clf6.fit(x_krr_train, y_krr_train)

plot_size = 0.001
x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

f2, axarr2 = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))

for idx, clf, tt in zip(product([0, 1], [0, 1, 2]),
                        [clf1, clf2, clf3, clf4, clf5, clf6],
                        ['Depth = {}'.format(n) for n in range(1, 7)]):

    y_range_predict = np.array([clf.predict_instance(x) for x in x_range]).reshape(-1, 1)
  
    axarr2[idx[0], idx[1]].plot(x_range, y_range_predict, color='r')
    axarr2[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
    axarr2[idx[0], idx[1]].set_title(tt)
    axarr2[idx[0], idx[1]].set_xlim(0, 1)
plt.show()

# %% [markdown]
# # Compare with scikit-learn for debugging

# %%

clf=DecisionTreeRegressor(criterion='absolute_error', max_depth=5, min_samples_split=3)
clf.fit(x_krr_train, y_krr_train)
# # Plotting decision regions
# x_min, x_max = x_train[:, 0].min() - 1, x_train[:, 0].max() + 1
# y_min, y_max = x_train[:, 1].min() - 1, x_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
#                      np.arange(y_min, y_max, 0.1))

# Z = np.array([clf.predict(x[np.newaxis,:]) for x in np.c_[xx.ravel(), yy.ravel()]])
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(x_train[:, 0], x_train[:, 1], 
# c=y_train_label[:,0], alpha=0.8)


plot_size = 0.001
x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

f2, axarr2 = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))
y_range_predict = clf.predict(x_range) 
axarr2[idx[0], idx[1]].plot(x_range, y_range_predict, color='r')
axarr2[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
axarr2[idx[0], idx[1]].set_title(tt)
axarr2[idx[0], idx[1]].set_xlim(0, 1)
plt.show()

# %% [markdown]
# # Gradient Boosting Method

# %%
#Pseudo-residual function.

def pseudo_residual_L2(train_target, train_predict):
    '''
    Compute the pseudo-residual based on current predicted value. 
    '''
    return train_target - train_predict

# %%
import functools
class gradient_boosting():
    '''
    Gradient Boosting regressor class
    :method fit: fitting model
    '''
    def __init__(self, n_estimator, pseudo_residual_func, learning_rate=0.01, 
                 min_sample=5, max_depth=5):
        '''
        Initialize gradient boosting class
        
        :param n_estimator: number of estimators (i.e. number of rounds of gradient boosting)
        :pseudo_residual_func: function used for computing pseudo-residual between training labels and predicted labels at each iteration
        :param learning_rate: step size of gradient descent
        '''
        self.n_estimator = n_estimator
        self.pseudo_residual_func = pseudo_residual_func
        self.learning_rate = learning_rate
        self.min_sample = min_sample
        self.max_depth = max_depth

        self.estimators = [] #will collect the n_estimator models
    def fit(self, train_data, train_target):
        '''
        Fit gradient boosting model
        :train_data array of inputs of size (n_samples, m_features) 
        :train_target array of outputs of size (n_samples,) 
        '''
        # Your code goes here 
        
        f_m=np.zeros(train_data.shape[0])
        for m in range(self.n_estimator):
            g_m = self.pseudo_residual_func(train_target, f_m )
            h_m = self.arg_min_at_m(g_m,train_data)
            f_m=f_m + h_m(train_data)*self.learning_rate 
            self.estimators.append(h_m)
        return self.estimators
    def arg_min_at_m(self, g_m, train_data):
        clf = DecisionTreeRegressor(criterion='squared_error', 
                                         min_samples_split=self.min_sample,
                                         max_depth=self.max_depth)
        clf.fit(train_data , g_m)
        h_f = lambda x: clf.predict(x)
        return h_f
    def predict(self, test_data):
        '''
        Predict value
        :train_data array of inputs of size (n_samples, m_features)
        '''
        # Your code goes here
        test_predict=np.zeros(test_data.shape[0])
        for func in self.estimators:
            test_predict=test_predict+func(test_data)*self.learning_rate
        return test_predict


# %% [markdown]
# # 1-D GBM visualization - KRR data

# %%
plot_size = 0.001
x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

f2, axarr2 = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))

for idx, i, tt in zip(product([0, 1], [0, 1, 2]),
                       [1, 5, 10, 20, 50, 100], 
                       ['n_estimator = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):
    
    gbm_1d = gradient_boosting(n_estimator=i, pseudo_residual_func=pseudo_residual_L2, 
                               max_depth=3, learning_rate=0.1)  
    gbm_1d.fit(x_krr_train, y_krr_train[:,0])
    
    y_range_predict = gbm_1d.predict(x_range)

    axarr2[idx[0], idx[1]].plot(x_range, y_range_predict, color='r')
    axarr2[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
    axarr2[idx[0], idx[1]].set_title(tt)
    axarr2[idx[0], idx[1]].set_xlim(0, 1)

# %%

plot_size = 0.001
x_range = np.arange(0., 1., plot_size).reshape(-1, 1)

f2, axarr2 = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 10))

for idx, i, tt in zip(product([0, 1], [0, 1, 2]),
                       [1, 5, 10, 20, 50, 100], 
                       ['n_estimator = {}'.format(n) for n in [1, 5, 10, 20, 50, 100]]):
    
    gbm_1d = GradientBoostingRegressor(learning_rate=.01, min_samples_split=5,n_estimators=i)
    gbm_1d.fit(x_krr_train, y_krr_train[:,0])
    
    y_range_predict = gbm_1d.predict(x_range)

    axarr2[idx[0], idx[1]].plot(x_range, y_range_predict, color='r')
    axarr2[idx[0], idx[1]].scatter(x_krr_train, y_krr_train, alpha=0.8)
    axarr2[idx[0], idx[1]].set_title(tt)
    axarr2[idx[0], idx[1]].set_xlim(0, 1)

# %% [markdown]
# # Sklearn implementation for Classification of images

# %%
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state


# %%
def pre_process_mnist_01():
    """
    Load the mnist datasets, selects the classes 0 and 1 
    and normalize the data.
    Args: none
    Outputs: 
        X_train: np.array of size (n_training_samples, n_features)
        X_test: np.array of size (n_test_samples, n_features)
        y_train: np.array of size (n_training_samples)
        y_test: np.array of size (n_test_samples)
    """
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1, 
                                    return_X_y=True, as_frame=False)
    indicator_01 = (y_mnist == '0') + (y_mnist == '1') 
    X_mnist_01 = X_mnist[indicator_01]
    y_mnist_01 = y_mnist[indicator_01]
    X_train, X_test, y_train, y_test = train_test_split(X_mnist_01, y_mnist_01,
                                                        test_size=0.33,
                                                        shuffle=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test)

    y_test = 2 * np.array([int(y) for y in y_test]) - 1
    y_train = 2 * np.array([int(y) for y in y_train]) - 1
    return X_train, X_test, y_train, y_test

# %%
X_train, X_test, y_train, y_test = pre_process_mnist_01()

# %%
def fit_and_eval(trials):
    train = np.zeros(  (len(trials) ))
    test = np.zeros((len(trials)))
    for trail_num in range(len(trials)):
        n=trials[trail_num]
        gbc = GradientBoostingClassifier(loss='deviance', max_depth=3,n_estimators=n)
        gbc.fit(X_train, y_train)
        score = lambda model, x, y : len(np.argwhere(model.predict(x)==y).flatten())/len(y)
        train[trail_num] = score(gbc,X_train, y_train)
        test[trail_num] = score(gbc,X_test, y_test)
    d= {'trails':np.array(trials), "train": train, "test": test}
    return d 
d = fit_and_eval([2, 5, 10, 100 , 200])
d

# %%
plt.plot(d["trails"], d["train"], label="training accuracy" )
plt.plot(d["trails"], d["test"], label="testing accuracy" )
plt.ylabel("accuracy")
plt.xlabel("number of estimators")
plt.title("number of estimators vs average train and test accuracy for Binomial Boost model")
plt.xscale('log')
plt.legend()
plt.show()

# %% [markdown]
# ## question 11. 

# %%

def fit_and_eval(trials):
    train = np.zeros(  (len(trials) ))
    test = np.zeros((len(trials)))
    for trail_num in range(len(trials)):
        n=trials[trail_num]
        rf_model = RandomForestClassifier(criterion='entropy', max_depth=3, n_estimators=n)
        rf_model.fit(X_train, y_train)
        score = lambda model, x, y : len(np.argwhere(model.predict(x)==y).flatten())/len(y)
        train[trail_num] = score(rf_model,X_train, y_train)
        test[trail_num] = score(rf_model,X_test, y_test)
    d= {'trails':np.array(trials), "train": train, "test": test}
    return d 
d_forest = fit_and_eval([2, 5, 10, 100 , 200])

# %%
plt.plot(d_forest["trails"], d_forest["train"], label="training accuracy" )
plt.plot(d_forest["trails"], d_forest["test"], label="testing accuracy" )
plt.ylabel("accuracy")
plt.xlabel("number of estimators")
plt.title("number of estimators vs train and test accuracy for Random Forest model")
plt.xscale('log')
plt.legend()
d_forest

# %%
print(max(d["test"]))
print(max(d_forest['test']))


