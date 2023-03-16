# %% [markdown]
# ## question 6 

# %%
from collections import Counter
from random import shuffle
from random import seed 
import os
import numpy as np
import matplotlib.pyplot as plt


# %%
def bag_of_words(list_of_wrods:list)->dict:
    """"
    input: list of strings (or string)
    output: dict containing word counts from list
    """
    return Counter(list_of_wrods)



# %% [markdown]
# ## question 7
# 

# %%
def folder_list(path,label):
    '''
    PARAMETER PATH IS THE PATH OF YOUR LOCAL FOLDER
    '''
    filelist = os.listdir(path)
    review = []
    for infile in filelist:
        file = os.path.join(path,infile)
        r = read_data(file)
        r.append(label)
        review.append(r)
    return review

def read_data(file):
    '''
    Read each file into a list of strings.
    Example:
    ["it's", 'a', 'curious', 'thing', "i've", 'found', 'that', 'when', 'willis', 'is', 'not', 'called', 'on',
    ...'to', 'carry', 'the', 'whole', 'movie', "he's", 'much', 'better', 'and', 'so', 'is', 'the', 'movie']
    '''
    f = open(file)
    lines = f.read().split(' ')
    symbols = '${}()[].,:;+-*/&|<>=~" '
    words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
    words = filter(None, words)
    return list(words)


def load_and_shuffle_data():
    '''
    pos_path is where you save positive review data.
    neg_path is where you save negative review data.
    '''
    pos_path = '/mnt/c/Users/buzga/Desktop/School/grad_school/spring_2023/machine_learning/homework/hw3/data/pos'
    neg_path = '/mnt/c/Users/buzga/Desktop/School/grad_school/spring_2023/machine_learning/homework/hw3/data/neg'

    pos_review = folder_list(pos_path,1)
    neg_review = folder_list(neg_path,-1)

    review = pos_review + neg_review
    shuffle(review)
    return review

# Taken from http://web.stanford.edu/class/cs221/ Assignment #2 Support Code
def dotProduct(d1, d2):
    """
    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in d2.items())

def increment(d1, scale, d2):
    """
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    """
    for f, v in d2.items():
        d1[f] = d1.get(f, 0) + v * scale
data=load_and_shuffle_data()

# %%
def train_test_split(data: list)->list:
    """"
    input: list of data from read data function
    output: X_train, X_validate: training,validation bag of words dicts, y_train, y_validate: training and validation labels 
    reshuffles data with a set seed for reproducabilty
    """
    rng=np.random.default_rng(15100873)
    rng.shuffle(data)
    train=data[:1500]
    X_train=[bag_of_words(review[:-1]) for review in train]
    y_train=[review[-1] for review in train]
    validate=data[1500:None]
    X_validate=[bag_of_words(review[:-1]) for review in validate]
    y_validate=[review[-1] for review in validate]
    return X_train, y_train, X_validate, y_validate
X_train, y_train, X_validate, y_validate=train_test_split(data)


# %% [markdown]
# ## question 8

# %%

def Pegasos_naive(X_train, y_train,epochs=5000, regularization_coef=.05):
    """
    input: 
    X_train: training data of shape (NxD) (each row is dictionary storign the word counts of that documnet )
    y_train: training labels of shape (Nx1) (array)
    epochs: number of epochs to run over (int)
    regularization_coef: lambda value used for normalization (float)
    output: 
    w: dictionary of length d with all words and asociated learned weights 
     
    """
    ## set up
    rng=np.random.default_rng(15100873)
    w_t=dict() ## initilize w
    #rng=np.random.default_rng(15100873) ## initilize random number generator with seed 
    curent_epoch=0 ##  epoch counter
    data=list(zip(X_train, y_train)) ## zip data togther to make easy to loop through


    while(curent_epoch<=epochs): 
        rng.shuffle(data) 
        t=0
        for x_i,y_i in data:
            t+=1
            eta_t=(1)/(regularization_coef*t)
            margin=y_i*dotProduct(w_t, x_i)
            scale=(-1)*eta_t*regularization_coef
            increment(d1=w_t, scale=scale, d2=w_t)
            if margin<1: ## go through each dictionary. if we get the reveiew wrong 
                scale=eta_t*y_i
                increment(d1=w_t, scale=scale, d2=x_i) ## we incremnt w_t meaning words in that document are now explicitly included in w_t
        curent_epoch+=1
        print(curent_epoch)
    return w_t
w_1=Pegasos_naive(X_train, y_train, epochs=10)


# %% [markdown]
# ## question 9

# %%

def Pegasos_better(X_train, y_train,epochs=5000, regularization_coef=.05):
    """
    input: 
    X_train: training data of shape (NxD) (each row is dictionary storign the word counts of that documnet )
    y_train: training labels of shape (Nx1) (array)
    epochs: number of epochs to run over (int)
    regularization_coef: lambda value used for normalization (float)
    output: 
    w: dictionary of length d with all words and asociated learned weights 
     
    """
    ## set up
    rng=np.random.default_rng(15100873)
    W_t=dict() ## which represnts 1/s_t times our independent output 
    curent_epoch=0 ##  epoch counter
    data=list(zip(X_train, y_train)) ## zip data togther to make easy to loop through
    while(curent_epoch<=epochs): 
        rng.shuffle(data) 
        t=1
        s_t=1 ## intiilize s_t
        for x_i,y_i in data:
            t+=1
            eta_t=(1)/(regularization_coef*t)
            margin=y_i*dotProduct(W_t, x_i)
            a=1-eta_t*regularization_coef
            s_t=(a)*s_t ## s_t updates  
            if margin<1:
                scale=(1/s_t)*eta_t*y_i
                increment(d1=W_t, scale=scale, d2=x_i) ##update w_t

        curent_epoch+=1
    w_t={key:W_t[key]*s_t for key in W_t.keys() } ## calulate intended value w_t
    return w_t
w_2=Pegasos_better(X_train, y_train, epochs=10)



# %%
for key in w_1.keys():
        if key in w_2.keys():
                print("word: {} \n,naive value={}\n better value={}".format(key,w_1[key], w_2[key]))
        else:
                print("word: {} not in both dictionaries\n naive value={}\n implicit better value=0".format(key, w_1[key]))

# %%
for key in w_2.keys():
        #if key in w_3.keys():
        print(key,w_2[key]==w_3[key])
        print("word: {} \n,naive value={}\n better value={}".format(key,w_2[key], w_3[key]))


# %% [markdown]
# ## question 11 

# %%
np.random.default_rng(15100873)
def classification_error(w, X, y):
    """
    input: 
    w: wieght dictionary of size d (effectily a DX1 vector)
    X: array of word counts for each document size (nXd)
    y: lables for each document Nx1 array
    output: 
    percentage of documents missclassfied using Xw to predict y.
    """
    np.random.default_rng(15100873)
    fixed_dot = lambda x: dotProduct(w,x)
    preds=np.array(list(map(fixed_dot, X)))
    margins=np.array(y)*preds
    return np.sum(margins<0)/len(margins) 
def test_error_over_alpha(min_alpha, max_alpha,X_train, y_train, X_test, y_test,granulairty=20, epochs=20):
    np.random.default_rng(15100873)
    out=[]
    plt_range=np.linspace(min_alpha, max_alpha,granulairty)
    for alpha in plt_range:
        w=Pegasos_better(X_train, y_train, epochs=epochs, regularization_coef=alpha)
        out.append(classification_error(w, X_test, y_test))
    plt.plot(plt_range, out)
    plt.xlabel("lambda value")
    plt.ylabel("zero-one erorr")
    plt.show()


test_error_over_alpha(min_alpha=.01, max_alpha=10, X_train=X_train,y_train=y_train, X_test=X_validate, y_test=y_validate,granulairty=20)
lambda_opt=7

# %%
test_error_over_alpha(min_alpha=3.5, max_alpha=8.5,  X_train=X_train,y_train=y_train, X_test=X_validate, y_test=y_validate,granulairty=20)


# %%
test_error_over_alpha(min_alpha=3.5, max_alpha=6,  X_train=X_train,y_train=y_train, X_test=X_validate, y_test=y_validate,granulairty=20)


# %%
test_error_over_alpha(min_alpha=4, max_alpha=5.5,  X_train=X_train,y_train=y_train, X_test=X_validate, y_test=y_validate,granulairty=20)


# %% [markdown]
# ## question 13

# %%
lambda_opt=5
def bin_preds_and_y(preds, y):
    preds_and_label=list(zip(preds,y))
    preds_and_label.sort(key=lambda w: w[0])
    out=[]
    cur=[[],[]]
    i=0
    j=1
    range_preds=max(preds)-min(preds)
    while(i<len(preds)):
        cur[0].append(preds_and_label[i][0])
        cur[1].append(preds_and_label[i][1])
        if(preds_and_label[i][0]>=(min(preds)+(j*(range_preds/10)))):
            out.append(np.array(cur))
            cur=[[],[]]
            j=j+1
        i+=1
    return out
def check_group_accuracy(outs):
    out_2=[[],[]]
    upper_bounds=[]
    lower_bounds=[]
    acurracy=[]
    for bin in outs:
        cor=np.correlate(bin[0],bin[1])
        b_1=np.array(bin[1])*np.array(bin[0])
        a=np.sum(b_1>0)/len(bin[1])
        b=max(bin[0])#, min(bin[0])]
        lower_bounds.append(min(bin[0]))
        upper_bounds.append(max(bin[0]))
        acurracy.append(a)
        #print("accuracy in range {1}-{0} is {2}".format(max(bin[0]), min(bin[0]), a))
        out_2[0].append(a)
        out_2[1].append(b)
    cor=np.corrcoef(np.abs(out_2[1]),out_2[0])[0,1]
    print("overall corelation between absolute value of score and accuracy is {0}".format(cor))
    return out_2, upper_bounds, lower_bounds, acurracy
w=Pegasos_better(X_train, y_train, epochs=5, regularization_coef=lambda_opt)
fixed_dot = lambda x: dotProduct(w,x)
preds=np.array(list(map(fixed_dot, X_validate)))
out=bin_preds_and_y(preds, y_validate)
out_2,upper_bounds, lower_bounds, acurracy=check_group_accuracy(out)
plt.plot(out_2[1], out_2[0])
plt.xlabel("score")
plt.ylabel("test accuracy")
plt.title("testing accuracy vs score linear classfication")
df=pd.DataFrame(data=lower_bounds, columns=["lower bound"])
df["upper bound"]=upper_bounds
df["test_accuracy"]=acurracy
df

# %% [markdown]
# ## question 14

# %%
import pandas as pd
def get_missclass_examples(X_train, X_validate ,y_train, y_validate, num=1):
    """"returns the n missclassfied examples on the validation set"""
    w=Pegasos_better(X_train, y_train, epochs=5, regularization_coef=lambda_opt)
    fixed_dot = lambda x: dotProduct(w,x)
    preds=np.array(list(map(fixed_dot, X_validate)))
    i=1
    for x,right in zip(X_validate,preds*y_validate): 
        if right<0 and i>=num:
            return  x,w
        elif right<0 and i<num:
            i=i+1
def feature_importance(x,w):
    out=dict()
    for key in x.keys():
        try:
            out[key]=(np.abs(x[key]*w[key]))
        except:
            out[key]=0
            w[key]=0
    return dict(sorted(out.items(), key=lambda item: item[1]))
lambda_opt=7
x,w=get_missclass_examples(X_train, X_validate ,y_train, y_validate)
features_important=feature_importance(x,w)
df=pd.DataFrame.from_dict(features_important, orient='index',columns=['|w_ix_i|'])
df["w_i"]=w
df["x_i"]=x
df.tail(7)

# %%
x,w=get_missclass_examples(X_train, X_validate ,y_train, y_validate, num=2)
features_important=feature_importance(x,w)
df=pd.DataFrame.from_dict(features_important, orient='index',columns=['|w_ix_i|'])
df["w_i"]=w
df["x_i"]=x

df.tail(7)


# %%
x,w=get_missclass_examples(X_train, X_validate ,y_train, y_validate, num=10)
features_important=feature_importance(x,w)
df=pd.DataFrame.from_dict(features_important, orient='index',columns=['|w_ix_i|'])
df["w_i"]=w
df["x_i"]=x
df.tail(7)





# %%
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.spatial
import functools
from  scipy.spatial.distance import cdist
from sklearn.metrics import zero_one_loss


# %% [markdown]
# ## question 21 

# %%
### Kernel function generators
def linear_kernel(X1, X2):
    """
    Computes the linear kernel between two sets of vectors.
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
    Returns:
        matrix of size n1xn2, with x1_i^T x2_j in position i,j
    """
    return np.dot(X1,np.transpose(X2))
 
def RBF_kernel(X1,X2,sigma):
    """
    Computes the RBF kernel between two sets of vectors   
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        sigma - the bandwidth (i.e. standard deviation) for the RBF/Gaussian kernel
    Returns:
        matrix of size n1xn2, with exp(-||x1_i-x2_j||^2/(2 sigma^2)) in position i,j
    """
    a=cdist(X1,X2,'sqeuclidean')*(-1)/(sigma**2)
    return np.exp(a)
    #TODO


def polynomial_kernel(X1, X2, offset, degree):
    """
    Computes the inhomogeneous polynomial kernel between two sets of vectors
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        offset, degree - two parameters for the kernel
    Returns:
        matrix of size n1xn2, with (offset + <x1_i,x2_j>)^degree in position i,j
    """
    #TODO
    return np.power((offset+np.dot(X1,np.transpose(X2))), degree)



# %% [markdown]
# # question 22 

# %%
x=np.array([-4,-1,0,2]).reshape(4,1)
linear_kernel(x,x)

# %% [markdown]
# ## question 23 

# %%
# PLot kernel machine functions

plot_step = .01
xpts = np.arange(-5.0, 6, plot_step).reshape(-1,1)
prototypes = np.array([-4,-1,0,2]).reshape(-1,1)

# Linear kernel
y = linear_kernel(prototypes, xpts) 
for i in range(len(prototypes)):
    label = "Linear@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.title("linear kernels")
plt.show() 


# PLot kernel machine functions


y = polynomial_kernel(prototypes, xpts,1,3) 
for i in range(len(prototypes)):
    label = "polynomial kernel@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.title("Polynomial kernel")
plt.show() 



y = RBF_kernel(prototypes, xpts, 1) 
for i in range(len(prototypes)):
    label = "RBF kernel@"+str(prototypes[i,:])
    plt.plot(xpts, y[i,:], label=label)
plt.legend(loc = 'best')
plt.title("RBF Kernel")
plt.show() 
 

 

# %%
class Kernel_Machine(object):
    def __init__(self, kernel, training_points, weights):
        """
        Args:
            kernel(X1,X2) - a function return the cross-kernel matrix between rows of X1 and rows of X2 for kernel k
            training_points - an nxd matrix with rows x_1,..., x_n
            weights - a vector of length n with entries alpha_1,...,alpha_n
        """

        self.kernel = kernel
        self.training_points = training_points
        self.weights = weights

    def predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  Specifically, jth entry of return vector is
                Sum_{i=1}^R alpha_i k(x_j, mu_i)
        """
        # TODO
        k_t=self.kernel(self.training_points,X)
        return np.dot(k_t.T, self.weights)

# %% [markdown]
# ## question 24 

# %%
prototypes = np.array([-1,0,1]).reshape(-1,1)
RBF=RBF_kernel(prototypes, prototypes, sigma=1)
weights=np.array([1,-1,1]).reshape(-1,1)
k = functools.partial(RBF_kernel, sigma=1)
poly_kernel=Kernel_Machine(kernel=k,weights=weights,training_points=prototypes)
x=np.array([2]).reshape(-1,1)
print(poly_kernel.predict(x))
x=np.linspace(-5,5,100)
out=[]
for i in x:
    out.append(poly_kernel.predict(np.array(i).reshape(1,1))[0])
out=np.array(out)
y=range(100)
plt.plot(x,out[y]) 
plt.xlabel("input value x")
plt.ylabel("predicted y value")
plt.title("example prediction for polynomial kernel")


# %% [markdown]
# Load train & test data; Convert to column vectors so it generalizes well to data in higher dimensions.

# %% [markdown]
# ## question 25

# %%
data_train,data_test = np.loadtxt("krr-train.txt"),np.loadtxt("krr-test.txt")
x_train, y_train = data_train[:,0].reshape(-1,1),data_train[:,1].reshape(-1,1)
x_test, y_test = data_test[:,0].reshape(-1,1),data_test[:,1].reshape(-1,1)

plt.scatter(x_train, y_train)
plt.title("training data ")

# %% [markdown]
# ## question 26 

# %%
def train_kernel_ridge_regression(X, y, kernel, l2reg):
    # TODO
    i=np.identity(X.shape[0])*l2reg
    k=kernel(X,X)
    alpha=np.dot((np.linalg.inv(i+k)),y)
    return Kernel_Machine(kernel, X, alpha)

# %% [markdown]
# ## Question 27

# %%
plot_step = .001
xpts = np.arange(0 , 1, plot_step).reshape(-1,1)
plt.plot(x_train,y_train,'o')
l2reg = 0.0001
for sigma in [.01,.1,1]:
    k = functools.partial(RBF_kernel, sigma=sigma)
    f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
    label = "Sigma="+str(sigma)+",L2Reg="+str(l2reg)
    plt.plot(xpts, f.predict(xpts), label=label)
plt.legend(loc = 'best')
plt.ylim(-1,1.5)
plt.show()

# %% [markdown]
# ## question 28

# %%
plot_step = .001
xpts = np.arange(0 , 1, plot_step).reshape(-1,1)
plt.plot(x_train,y_train,'o')
sigma= .02
for l2reg in [.0001,.01,.1,2]:
    k = functools.partial(RBF_kernel, sigma=sigma)
    f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
    label = "Sigma="+str(sigma)+",L2Reg="+str(l2reg)
    plt.plot(xpts, f.predict(xpts), label=label)
plt.legend(loc = 'best')
plt.ylim(-1,1.5)
plt.show()

# %% [markdown]
# ## question 29

# %%
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class KernelRidgeRegression(BaseEstimator, RegressorMixin):  
    """sklearn wrapper for our kernel ridge regression"""
     
    def __init__(self, kernel="RBF", sigma=1, degree=2, offset=1, l2reg=1):        
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.offset = offset
        self.l2reg = l2reg 

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        if (self.kernel == "linear"):
            self.k = linear_kernel
        elif (self.kernel == "RBF"):
            self.k = functools.partial(RBF_kernel, sigma=self.sigma)
        elif (self.kernel == "polynomial"):
            self.k = functools.partial(polynomial_kernel, offset=self.offset, degree=self.degree)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        
        self.kernel_machine_ = train_kernel_ridge_regression(X, y, self.k, self.l2reg)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return(self.kernel_machine_.predict(X))

    def score(self, X, y=None):
        # get the average square error
        return(((self.predict(X)-y)**2).mean()) 

# %%
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error,make_scorer
import pandas as pd
    
test_fold = [-1]*len(x_train) + [0]*len(x_test)   #0 corresponds to test, -1 to train
predefined_split = PredefinedSplit(test_fold=test_fold)

# %%
param_grid = [{'kernel': ['RBF'],'sigma':[.1,1,10], 'l2reg': np.exp2(-np.arange(-5,5,1))},
              {'kernel':['polynomial'],'offset':[-1,0,1], 'degree':[2,3,4],'l2reg':[10, .1, .01] },
              {'kernel':['linear'],'l2reg': [10,1,.01]}]
kernel_ridge_regression_estimator = KernelRidgeRegression()
grid = GridSearchCV(kernel_ridge_regression_estimator, 
                    param_grid,
                    cv = predefined_split,
                    scoring = make_scorer(mean_squared_error,greater_is_better = False),
                    return_train_score=True
                   )
grid.fit(np.vstack((x_train,x_test)),np.vstack((y_train,y_test))) 

# %%
pd.set_option('display.max_rows', 20)
df = pd.DataFrame(grid.cv_results_)
# Flip sign of score back, because GridSearchCV likes to maximize,
# so it flips the sign of the score if "greater_is_better=FALSE"
df['mean_test_score'] = -df['mean_test_score']
df['mean_train_score'] = -df['mean_train_score']
cols_to_keep = ["param_degree", "param_kernel","param_l2reg" ,"param_offset","param_sigma",
        "mean_test_score","mean_train_score"]
df_toshow = df[cols_to_keep].fillna('-')
df_to_show=df_toshow.sort_values(by=["mean_test_score"])

# %%
df_to_show[df_to_show['param_kernel']=="RBF"]

# %%
df_to_show[df_to_show['param_kernel']=="linear"]

# %%
df_to_show[df_to_show['param_kernel']=="polynomial"]

# %% [markdown]
# ## question 30

# %%
## Plot the best polynomial and RBF fits you found
plot_step = .01
xpts = np.arange(-.5 , 1.5, plot_step).reshape(-1,1)
plt.plot(x_train,y_train,'o')
#Plot best polynomial fit
offset= -1
degree = 4
l2reg = 0.1000
k = functools.partial(polynomial_kernel, offset=offset, degree=degree)
f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
label = "Offset="+str(offset)+",Degree="+str(degree)+",L2Reg="+str(l2reg)
plt.plot(xpts, f.predict(xpts), label=label)
#Plot best RBF fit
sigma = 0.1
l2reg= 0.0625
k = functools.partial(RBF_kernel, sigma=sigma)
f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
label = "Sigma="+str(sigma)+",L2Reg="+str(l2reg)
plt.plot(xpts, f.predict(xpts), label=label)
plt.legend(loc = 'best')
plt.ylim(-1,1.75)
plt.title("best RBF and Polynomial kernel found")
plt.show()

# %% [markdown]
# # Kernel SVM optional problem

# %%
# Load and plot the SVM data
#load the training and test sets
data_train,data_test = np.loadtxt("svm-train.txt"),np.loadtxt("svm-test.txt")
x_train, y_train = data_train[:,0:2], data_train[:,2].reshape(-1,1)
x_test, y_test = data_test[:,0:2], data_test[:,2].reshape(-1,1)

#determine predictions for the training set
yplus = np.ma.masked_where(y_train[:,0]<=0, y_train[:,0])
xplus = x_train[~np.array(yplus.mask)]
yminus = np.ma.masked_where(y_train[:,0]>0, y_train[:,0])
xminus = x_train[~np.array(yminus.mask)]

#plot the predictions for the training set
figsize = plt.figaspect(1)
f, (ax) = plt.subplots(1, 1, figsize=figsize) 

pluses = ax.scatter (xplus[:,0], xplus[:,1], marker='+', c='r', label = '+1 labels for training set')
minuses = ax.scatter (xminus[:,0], xminus[:,1], marker=r'$-$', c='b', label = '-1 labels for training set')

ax.set_ylabel(r"$x_2$", fontsize=11)
ax.set_xlabel(r"$x_1$", fontsize=11)
ax.set_title('Training set size = %s'% len(data_train), fontsize=9)  
ax.axis('tight')
ax.legend(handles=[pluses, minuses], fontsize=9)
plt.show()

# %%
from collections import Counter
from random import shuffle
from random import seed 
import os
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## question 33

# %%
def train_soft_svm(X_train, y_train,kernel,epochs=100, regularization_coef=.05):
    ## set up
    W_t=np.zeros((X_train.shape[0],1)) ## which represnts 1/s_t our independent output
    seed(15100873) ## set random seed 
    curent_epoch=0 ##  epoch counter
    data=list(zip(X_train, y_train)) ## zip data togther to make easy to loop through
    k=kernel(X_train, X_train)
    while(curent_epoch<=epochs): 
        shuffle(data)
        t=1
        s_t=1 ## intiilize s_5
        for x_i,y_i in data:
            x_i=x_i.reshape(-1,1)
            t+=1
            eta_t=(1)/(regularization_coef*t)
            a=1-eta_t*regularization_coef
            s_t=(a)*s_t ## s_t updates
            k_t=kernel(X_train,x_i.T)
            pred=np.dot(k_t.T, W_t)
            margin=y_i*pred  

            if margin<1:
                scale=(1/s_t)*eta_t*y_i
                W_t=np.add(W_t,scale*(np.dot( k.T,y_train)))
 
        curent_epoch+=1

    return Kernel_Machine(kernel, X_train, W_t*s_t)


from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class soft_svm(BaseEstimator, RegressorMixin):  
    """sklearn wrapper for our kernel ridge regression"""
     
    def __init__(self, kernel="RBF", sigma=1, degree=2, offset=1, l2reg=1):        
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.offset = offset
        self.l2reg = l2reg 

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        if (self.kernel == "linear"):
            self.k = linear_kernel
        elif (self.kernel == "RBF"):
            self.k = functools.partial(RBF_kernel, sigma=self.sigma)
        elif (self.kernel == "polynomial"):
            self.k = functools.partial(polynomial_kernel, offset=self.offset, degree=self.degree)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        
        self.kernel_machine_ = train_soft_svm(X, y, self.k, self.l2reg)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return(self.kernel_machine_.predict(X))

    def score(self, X, y=None):
        # get the average square error
        return(((self.predict(X)-y)**2).mean()) 

# %% [markdown]
# ## question 34

# %%
# Code to help plot the decision regions
# (Note: This ode isn't necessarily entirely appropriate for the questions asked. So think about what you are doing.)
 
# sigma=1
k = functools.partial(RBF_kernel, sigma=sigma)
f = train_soft_svm(x_train, y_train, k)
x_train.shape
#determine the decision regions for the predictions
x1_min = min(x_test[:,0])
x1_max= max(x_test[:,0])
x2_min = min(x_test[:,1])
x2_max= max(x_test[:,1])
h=0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                     np.arange(x2_min, x2_max, h))

Z = f.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#determine the predictions for the test set
y_bar = f.predict (x_test) 
yplus = np.ma.masked_where(y_bar<=0, y_bar)
xplus = x_test[[i for i in range(len(x_test)) if ~np.array(yplus.mask)[i]]]

yminus = np.ma.masked_where(y_bar>0, y_bar)
xminus = x_test[[i for i in range(len(x_test)) if ~np.array(yminus.mask)[i]]]
# #plot the learned boundary and the predictions for the test set
figsize = plt.figaspect(1)
f, (ax) = plt.subplots(1, 1, figsize=figsize) 
decision =ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
pluses = ax.scatter (xplus[:,0], xplus[:,1], marker='+', c='r', label = '+1 prediction for test set')
minuses = ax.scatter (xminus[:,0], xminus[:,1], marker=r'$-$', c='b', label = '-1 prediction for test set')
ax.set_ylabel(r"$x_2$", fontsize=11)
ax.set_xlabel(r"$x_1$", fontsize=11)
ax.set_title('SVM with RBF Kernel: training set size = %s'% len(data_train), fontsize=9)  
ax.axis('tight')
ax.legend(handles=[pluses, minuses], fontsize=9)
plt.show()

np.unique(y_train)
np.unique(y_test)

# %%
param_grid = [{'kernel': ['RBF'],'sigma':[.1,1,10], 'l2reg': np.exp2(-np.arange(-5,5,1))},
              {'kernel':['polynomial'],'offset':[-1,0,1], 'degree':[2,3,4],'l2reg':[10, .1, .01] },
              {'kernel':['linear'],'l2reg': [10,1,.01]}]
soft_svm_regression_estimator = soft_svm()
grid = GridSearchCV(soft_svm_regression_estimator, 
                    param_grid,
                    cv = predefined_split,
                    scoring = make_scorer(sklearn.metrics.hinge_loss,greater_is_better = False),
                    return_train_score=True
                   )
grid.fit(np.vstack((x_train,x_test)),np.vstack((y_train,y_test))) 

# %%
pd.set_option('display.max_rows', 20)
df = pd.DataFrame(grid.cv_results_)
# Flip sign of score back, because GridSearchCV likes to maximize,
# so it flips the sign of the score if "greater_is_better=FALSE"
df['mean_test_score'] = -df['mean_test_score']
df['mean_train_score'] = -df['mean_train_score']
cols_to_keep = ["param_degree", "param_kernel","param_l2reg" ,"param_offset","param_sigma",
        "mean_test_score","mean_train_score"]
df_toshow = df[cols_to_keep].fillna('-')
df_toshow=df_toshow.sort_values(by=["mean_test_score"])




# %%
df_toshow[df_toshow["param_kernel"]=="RBF"]

# %%
df_toshow[df_toshow["param_kernel"]=="linear"]

# %%
df_toshow[df_toshow["param_kernel"]=="polynomial"]

# %% [markdown]
# ## question 35

# %%
# Code to help plot the decision regions
# (Note: This ode isn't necessarily entirely appropriate for the questions asked. So think about what you are doing.)
kernel=df_toshow[df_toshow["param_kernel"]=="linear"]["param_kernel"].iloc[0]
l2=df_toshow[df_toshow["param_kernel"]=="linear"]["param_l2reg"].iloc[0]
print(df_toshow[df_toshow["param_kernel"]=="linear"]["mean_test_score"].iloc[0])
f=soft_svm(kernel=kernel, l2reg=l2)
f.fit(x_train, y_train)

#determine the decision regions for the predictions
x1_min = min(x_test[:,0])
x1_max= max(x_test[:,0])
x2_min = min(x_test[:,1])
x2_max= max(x_test[:,1])
h=0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                     np.arange(x2_min, x2_max, h))

Z = f.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#determine the predictions for the test set
y_bar = f.predict (x_test) 
yplus = np.ma.masked_where(y_bar<=0, y_bar)
xplus = x_test[[i for i in range(len(x_test)) if ~np.array(yplus.mask)[i]]]

yminus = np.ma.masked_where(y_bar>0, y_bar)
xminus = x_test[[i for i in range(len(x_test)) if ~np.array(yminus.mask)[i]]]
# #plot the learned boundary and the predictions for the test set
figsize = plt.figaspect(1)
f, (ax) = plt.subplots(1, 1, figsize=figsize) 
decision =ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
pluses = ax.scatter (xplus[:,0], xplus[:,1], marker='+', c='r', label = '+1 prediction for test set')
minuses = ax.scatter (xminus[:,0], xminus[:,1], marker=r'$-$', c='b', label = '-1 prediction for test set')
ax.set_ylabel(r"$x_2$", fontsize=11)
ax.set_xlabel(r"$x_1$", fontsize=11)
ax.set_title('best SVM with Linear Kernel: training set size = %s'% len(data_train), fontsize=9)  
ax.axis('tight')
ax.legend(handles=[pluses, minuses], fontsize=9)
plt.show()

np.unique(y_train)
np.unique(y_test)

# %%
# Code to help plot the decision regions
# (Note: This ode isn't necessarily entirely appropriate for the questions asked. So think about what you are doing.)
kernel=df_toshow[df_toshow["param_kernel"]=="RBF"]["param_kernel"].iloc[0]
l2=df_toshow[df_toshow["param_kernel"]=="RBF"]["param_l2reg"].iloc[0]
sigma=df_toshow[df_toshow["param_kernel"]=="RBF"]["param_sigma"].iloc[0]

# print(df_toshow[df_toshow["param_kernel"]=="RBF"]["mean_test_score"].iloc[0])
f=soft_svm(kernel=kernel, sigma=sigma, l2reg=l2)
f.fit(x_train, y_train)

#determine the decision regions for the predictions
x1_min = min(x_test[:,0])
x1_max= max(x_test[:,0])
x2_min = min(x_test[:,1])
x2_max= max(x_test[:,1])
h=0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                     np.arange(x2_min, x2_max, h))

Z = f.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#determine the predictions for the test set
y_bar = f.predict (x_test) 
yplus = np.ma.masked_where(y_bar<=0, y_bar)
xplus = x_test[[i for i in range(len(x_test)) if ~np.array(yplus.mask)[i]]]

yminus = np.ma.masked_where(y_bar>0, y_bar)
xminus = x_test[[i for i in range(len(x_test)) if ~np.array(yminus.mask)[i]]]
# #plot the learned boundary and the predictions for the test set
figsize = plt.figaspect(1)
f, (ax) = plt.subplots(1, 1, figsize=figsize) 
decision =ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
pluses = ax.scatter (xplus[:,0], xplus[:,1], marker='+', c='r', label = '+1 prediction for test set')
minuses = ax.scatter (xminus[:,0], xminus[:,1], marker=r'$-$', c='b', label = '-1 prediction for test set')
ax.set_ylabel(r"$x_2$", fontsize=11)
ax.set_xlabel(r"$x_1$", fontsize=11)
ax.set_title('best SVM with RBF Kernel: training set size = %s'% len(data_train), fontsize=9)  
ax.axis('tight')
ax.legend(handles=[pluses, minuses], fontsize=9)
plt.show()

# %%
# Code to help plot the decision regions
# (Note: This ode isn't necessarily entirely appropriate for the questions asked. So think about what you are doing.)
kernel=df_toshow[df_toshow["param_kernel"]=="polynomial"]["param_kernel"].iloc[0]
l2=df_toshow[df_toshow["param_kernel"]=="polynomial"]["param_l2reg"].iloc[0]
offset=df_toshow[df_toshow["param_kernel"]=="polynomial"]["param_offset"].iloc[0]
degree=df_toshow[df_toshow["param_kernel"]=="polynomial"]["param_degree"].iloc[0]
f=soft_svm(kernel=kernel, l2reg=l2, offset=offset, degree=degree)
f.fit(x_train, y_train)
print(kernel,l2, offset,degree)

#determine the decision regions for the predictions
x1_min = min(x_test[:,0])
x1_max= max(x_test[:,0])
x2_min = min(x_test[:,1])
x2_max= max(x_test[:,1])
h=0.1
xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                     np.arange(x2_min, x2_max, h))

Z = f.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

#determine the predictions for the test set
y_bar = f.predict (x_test) 
yplus = np.ma.masked_where(y_bar<=0, y_bar)
xplus = x_test[[i for i in range(len(x_test)) if ~np.array(yplus.mask)[i]]]

yminus = np.ma.masked_where(y_bar>0, y_bar)

xminus = x_test[[i for i in range(len(x_test)) if ~np.array(yminus.mask)]]
# #plot the learned boundary and the predictions for the test set
figsize = plt.figaspect(1)
f, (ax) = plt.subplots(1, 1, figsize=figsize) 
decision =ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
pluses = ax.scatter (xplus[:,0], xplus[:,1], marker='+', c='r', label = '+1 prediction for test set')
minuses = ax.scatter (xminus[:,0], xminus[:,1], marker=r'$-$', c='b', label = '-1 prediction for test set')
ax.set_ylabel(r"$x_2$", fontsize=11)
ax.set_xlabel(r"$x_1$", fontsize=11)
ax.set_title('best SVM with polynomial Kernel: training set size = %s'% len(data_train), fontsize=9)  
ax.axis('tight')
ax.legend(handles=[pluses, minuses], fontsize=9)
plt.show()



----------------

def Pegasos_naive(X_train, y_train,epochs=5000, regularization_coef=.05):
    """
    input: 
    X_train: training data of shape (NxD) (each row is dictionary storign the word counts of that documnet )
    y_train: training labels of shape (Nx1) (array)
    epochs: number of epochs to run over (int)
    regularization_coef: lambda value used for normalization (float)
    output: 
    w: dictionary of length d with all words and asociated learned weights 
     
    """
    ## set up
    rng=np.random.default_rng(15100873)
    w_t=dict() ## initilize w
    #rng=np.random.default_rng(15100873) ## initilize random number generator with seed 
    curent_epoch=0 ##  epoch counter
    data=list(zip(X_train, y_train)) ## zip data togther to make easy to loop through
    t=0
    while(curent_epoch<=epochs): 
        rng.shuffle(data) 
        for x_i,y_i in data:
            t+=1
            eta_t=(1)/(regularization_coef*t)
            margin=y_i*dotProduct(w_t, x_i)
            scale=(-1)*eta_t*regularization_coef
            increment(d1=w_t, scale=scale, d2=w_t)
            if margin<1: ## go through each dictionary. if we get the reveiew wrong 
                scale=eta_t*y_i
                increment(d1=w_t, scale=scale, d2=x_i) ## we incremnt w_t meaning words in that document are now explicitly included in w_t
        curent_epoch+=1
        print(curent_epoch)
    return w_t
w_1=Pegasos_naive(X_train, y_train, epochs=10)



def Pegasos_better(X_train, y_train,epochs=5000, regularization_coef=.05):
    """
    input: 
    X_train: training data of shape (NxD) (each row is dictionary storign the word counts of that documnet )
    y_train: training labels of shape (Nx1) (array)
    epochs: number of epochs to run over (int)
    regularization_coef: lambda value used for normalization (float)
    output: 
    w: dictionary of length d with all words and asociated learned weights 
     
    """
    ## set up
    rng=np.random.default_rng(15100873)
    W_t=dict() ## which represnts 1/s_t times our independent output 
    curent_epoch=0 ##  epoch counter
    data=list(zip(X_train, y_train)) ## zip data togther to make easy to loop through
    t=1
    s_t=1 ## intiilize s_t
    while(curent_epoch<=epochs): 
        rng.shuffle(data) 
        for x_i,y_i in data:
            t=t+1
            eta_t=(1)/(regularization_coef*t)
            margin=y_i*dotProduct(W_t, x_i)
            a=1-eta_t*regularization_coef
            s_t=(a)*s_t ## s_t updates  
            if margin<1:
                scale=(1/s_t)*eta_t*y_i
                increment(d1=W_t, scale=scale, d2=x_i) ##update w_t
        curent_epoch+=1
    w_t={key:W_t[key]*s_t for key in W_t.keys() } ## calulate intended value w_t
    return w_t
w_2=Pegasos_better(X_train, y_train, epochs=10)

def RBF_kernel(X1,X2,sigma):
    """
    Computes the RBF kernel between two sets of vectors   
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        sigma - the bandwidth (i.e. standard deviation) for the RBF/Gaussian kernel
    Returns:
        matrix of size n1xn2, with exp(-||x1_i-x2_j||^2/(2 sigma^2)) in position i,j
    """
    a=cdist(X1,X2,'sqeuclidean')*(-1)/(2*sigma**2)
    return np.exp(a)
    #TODO


def polynomial_kernel(X1, X2, offset, degree):
    """
    Computes the inhomogeneous polynomial kernel between two sets of vectors
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        offset, degree - two parameters for the kernel
    Returns:
        matrix of size n1xn2, with (offset + <x1_i,x2_j>)^degree in position i,j
    """
    #TODO
    return np.power((offset+np.dot(X1,np.transpose(X2))), degree)

def train_soft_svm(X_train, y_train,kernel,epochs=100, regularization_coef=.05):
    ## set up
    W_t=np.zeros((X_train.shape[0],1)) ## which represnts 1/s_t our independent output
    seed(15100873) ## set random seed 
    curent_epoch=0 ##  epoch counter
    data=list(zip(X_train, y_train)) ## zip data togther to make easy to loop through
    k=kernel(X_train, X_train)
    t=1
    s_t=1 ## intiilize s_5
    while(curent_epoch<=epochs): 
        shuffle(data)
        for x_i,y_i in data:
            x_i=x_i.reshape(-1,1)
            t+=1
            eta_t=(1)/(regularization_coef*t)
            a=1-eta_t*regularization_coef
            s_t=(a)*s_t ## s_t updates
            k_t=kernel(X_train,x_i.T)
            pred=np.dot(k_t.T, W_t)
            margin=y_i*pred  

            if margin<1:
                scale=(1/s_t)*eta_t*y_i
                W_t=np.add(W_t,scale*(np.dot( k.T,y_train)))
 
        curent_epoch+=1

    return Kernel_Machine(kernel, X_train, W_t*s_t)


from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin

class soft_svm(BaseEstimator, RegressorMixin):  
    """sklearn wrapper for our kernel ridge regression"""
     
    def __init__(self, kernel="RBF", sigma=1, degree=2, offset=1, l2reg=1):        
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.offset = offset
        self.l2reg = l2reg 

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """
        if (self.kernel == "linear"):
            self.k = linear_kernel
        elif (self.kernel == "RBF"):
            self.k = functools.partial(RBF_kernel, sigma=self.sigma)
        elif (self.kernel == "polynomial"):
            self.k = functools.partial(polynomial_kernel, offset=self.offset, degree=self.degree)
        else:
            raise ValueError('Unrecognized kernel type requested.')
        
        self.kernel_machine_ = train_soft_svm(X, y, self.k, self.l2reg)

        return self

    def predict(self, X, y=None):
        try:
            getattr(self, "kernel_machine_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        return(self.kernel_machine_.predict(X))

    def score(self, X, y=None):
        # get the average square error
        return(((self.predict(X)-y)**2).mean()) 