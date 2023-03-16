import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
data=pd.read_csv('/mnt/c/Users/buzga/Desktop/School/grad_school/spring_2023/machine_learning/homework/hw2/hw2/ridge_regression_dataset.csv')

# print(  [ col  for col in list(data.columns) if( not np.all(data[col]==data[col][0]) )] )

#######################################
### Feature normalization
def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size(num_instances, num_features)
        test - test set, a 2D numpy array of size(num_instances, num_features)

    Returns:
        train_normalized - training set after normalization
        test_normalized - test set after normalization
    """
    # TODO
    ## finds non constant rows 
    train_vals=[i for i in range(len(train[0,:])) if (np.all(train == train[0,:], axis = 0)[i]==False)] 
    test_vals=[i for i in range(len(test[0,:])) if (np.all(test == test[0,:], axis = 0)[i]==False)]
    ## standardizes across non_constant rows. and discards constnat features. 
    train_numerator=(train[:,train_vals]- np.min(train[:,train_vals],axis=0))
    train_denominator=(np.max(train[:,train_vals],axis=0)- np.min(train[:,train_vals],axis=0))
    test_numeroatro=(test[:,test_vals]- np.min(test[:,test_vals],axis=0))
    test_denomiator=(np.max(test[:,test_vals],axis=0)- np.min(test[:,test_vals],axis=0))
    return train_numerator/train_denominator, test_numeroatro/test_denomiator


#######################################
### The square loss function
def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the average square loss for predicting y with X*theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D array of size(num_features)

    Returns:
        loss - the average square loss, scalar
    """
    #TODO
    a=(X@theta-y)
    return (a.T@a)/y.shape[0]


def compute_regularized_square_loss(X,y,theta,lambda_reg):
    a=(X@theta-y)
    return (a.T@a)/y.shape[0]+ lambda_reg*(theta.T@theta)
#######################################
### The gradient of the square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute the gradient of the average square loss(as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    #TODO
    a=(X@theta)
    b=a-y
    return (2/y.shape[0])*X.T@b
#######################################
### Gradient checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm. Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
(e_1 =(1,0,0,...,0), e_2 =(0,1,0,...,0), ..., e_d =(0,...,0,1))

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
(J(theta + epsilon * e_i) - J(theta - epsilon * e_i)) /(2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicating whether the gradient is correct or not
    """
    #TODO
    true_gradient = compute_square_loss_gradient(X, y, theta) #The true gradient
    #print("true gradinet is ", true_gradient)
    num_features = theta.shape[0]
    # approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    approx_grad= list(map(lambda I:( compute_square_loss(X,y,theta + epsilon * I)-compute_square_loss(X,y,theta - epsilon * I) )/(2*epsilon),np.identity(num_features)))
    #print("print estimated gradient is", approx_grad)
    distances=true_gradient-approx_grad
    euclidian_distance=(distances.T)@(distances)
    #print("there euclidian distance is, ", euclidian_distance)
    result=euclidian_distance<=tolerance
    #print("thus is is {0} that the true and aproximate gradients are within tollerence".format(result))
    return euclidian_distance<=tolerance
    
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def regularized_grad_checker(X, y, theta,lambda_reg, epsilon=0.01, tolerance=1e-4):
    true_gradient =  compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
    #print("true gradinet is ", true_gradient)
    num_features = theta.shape[0]
    # approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    approx_grad= list(map(lambda I:( compute_regularized_square_loss(X,y,theta + epsilon * I,lambda_reg)-compute_regularized_square_loss(X,y,theta - epsilon * I, lambda_reg) )/(2*epsilon),np.identity(num_features)))
    #print("print estimated gradient is", approx_grad)
    distances=true_gradient-approx_grad
    euclidian_distance=(distances.T)@(distances)
    #print("there euclidian distance is, ", euclidian_distance)
    result=euclidian_distance<=tolerance
    #print(result)
    #print("thus is is {0} that the true and aproximate gradients are within tollerence".format(result))
    return euclidian_distance<=tolerance




#######################################
### Generic gradient checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, 
                             epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. 
    And check whether gradient_func(X, y, theta) returned the true 
    gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO
    true_gradient = gradient_func(X, y, theta) #The true gradient
    #print("true gradinet is ", true_gradient)
    num_features = theta.shape[0]
    # approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    approx_grad= list(map(lambda I:( objective_func(X,y,theta + epsilon * I)-objective_func(X,y,theta - epsilon * I) )/(2*epsilon),np.identity(num_features)))
    #print("print estimated gradient is", approx_grad)
    distances=true_gradient-approx_grad
    euclidian_distance=(distances.T)@(distances)
    #print("there euclidian distance is, ", euclidian_distance)
    result=euclidian_distance<=tolerance
    #print("thus is is {0} that the true and aproximate gradients are within tollerence".format(result))
    return euclidian_distance<=tolerance





#######################################
### Batch gradient descent
def batch_grad_descent(X, y, alpha=0.1, num_step=1000, grad_check=False):
    """
    In this question you will implement batch gradient descent to
    minimize the average square loss objective.

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array,(num_step+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  #Initialize loss_hist
    theta = np.zeros(num_features)  #Initialize theta
    i=0
    while i<num_step:
        if(grad_check):
            if(generic_gradient_checker(X,y,theta,objective_func=compute_square_loss, gradient_func=compute_square_loss_gradient )):
                current_grad=compute_square_loss_gradient(X, y, theta)
            else:
                print("htt")
                epsilon=0.01
                #current_grad=np.array(list(map(lambda I:( compute_square_loss(X,y,theta + epsilon * I)-compute_square_loss(X,y,theta - epsilon * I) )/(2*epsilon),np.identity(num_features))))
        else:
            current_grad=compute_square_loss_gradient(X, y, theta)
        theta=theta-(alpha*current_grad)
        theta_hist[i+1]=theta
        loss_hist[i+1]=compute_square_loss(X,y,theta)
        i=i+1

    return theta_hist,loss_hist

    #TODO
def test_loss_for_batch_grad_descent(X_train, y_train,X_test,y_test, alpha=0.1, num_step=1000, grad_check=False):
    """

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array,(num_step+1)
    """
    i=0
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  #Initialize loss_hist
    theta = np.zeros(num_features)  #Initialize theta
    while i<num_step:
            if(grad_check):
                if(generic_gradient_checker(X_train,y_train,theta,objective_func=compute_square_loss, gradient_func=compute_square_loss_gradient )):
                    current_grad=compute_square_loss_gradient(X_train,y_train, theta)
                else:
                    print("hit")
                    epsilon=0.01
                    #current_grad=np.array(list(map(lambda I:( compute_square_loss(X_train,y_train,theta + epsilon * I)-compute_square_loss(X_train,y_train,theta - epsilon * I) )/(2*epsilon),np.identity(num_features))))
            else:
                current_grad=compute_square_loss_gradient(X_train,y_train, theta)
            theta=theta-(alpha*current_grad)
            theta_hist[i+1]=theta
            loss_hist[i+1]=compute_square_loss(X_test,y_test,theta)
            i=i+1
    return theta_hist,loss_hist


#######################################
### The gradient of regularized batch gradient descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized average square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        theta - the parameter vector, 1D numpy array of size(num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size(num_features)
    """
    #TODO
    A=(X@theta - y)
    return (2/y.shape[0])*(X.T@A)+2*lambda_reg*(theta)


#######################################
### Regularized batch gradient descent
def regularized_grad_descent(X, y, alpha=0.05, lambda_reg=10**-2, num_step=1000) :
    """
    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        num_step - number of steps to run
    
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step+1) is theta_hist[-1]
        loss hist - the history of average square loss function without the regularization term, 1D numpy array.
    """

    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_step+1, num_features)) #Initialize theta_hist
    loss_hist = np.zeros(num_step+1) #Initialize loss_hist
    for i in range(num_step):
        if(regularized_grad_checker(X,y,theta,lambda_reg)):
            grad=compute_regularized_square_loss_gradient(X,y,theta, lambda_reg)
        else:
            epsilon=0.01
            grad= list(map(lambda I:( compute_regularized_square_loss(X,y,theta + epsilon * I,lambda_reg)-compute_regularized_square_loss(X,y,theta - epsilon * I, lambda_reg) )/(2*epsilon),np.identity(num_features)))
            print(grad)
        #grad=compute_regularized_square_loss_gradient(X,y,theta, lambda_reg)
        theta=theta-alpha*grad
        theta_hist[i+1]=theta
        loss_hist[i+1]=compute_square_loss(X,y,theta)
    return theta_hist, loss_hist

def test_loss_for_regularized_grad_descent(X_train, y_train,X_test,y_test, alpha=0.1, num_step=1000, grad_check=False,lambda_reg=10**-2):
    """

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - step size in gradient descent
        num_step - number of steps to run
        grad_check - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size(num_step+1, num_features)
                     for instance, theta in step 0 should be theta_hist[0], theta in step(num_step) is theta_hist[-1]
        loss_hist - the history of average square loss on the data, 1D numpy array,(num_step+1)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_step + 1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_step + 1)  #Initialize loss_hist
    theta = np.zeros(num_features)  #Initialize theta
    for i in range(num_step):
        current_grad=compute_regularized_square_loss_gradient(X_train, y_train, theta, lambda_reg)
        theta=theta-(alpha*current_grad)
        theta_hist[i+1]=theta
        loss_hist[i+1]=compute_square_loss(X_test,y_test,theta)

    return theta_hist,loss_hist





#######################################
### Stochastic gradient descent
def stochastic_grad_descent(X, y, alpha=0.01, lambda_reg=10**-2, num_epoch=1000, eta0=False):
    """
    In this question you will implement stochastic gradient descent with regularization term

    Args:
        X - the feature vector, 2D numpy array of size(num_instances, num_features)
        y - the label vector, 1D numpy array of size(num_instances)
        alpha - string or float, step size in gradient descent
                NOTE: In SGD, it's not a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every step is the float.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t).
                if alpha == "1/t", alpha = 1/t.
        lambda_reg - the regularization coefficient
        num_epoch - number of epochs to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size(num_epoch, num_instances, num_features)
                     for instance, theta in epoch 0 should be theta_hist[0], theta in epoch(num_epoch) is theta_hist[-1]
        loss hist - the history of loss function vector, 2D numpy array of size(num_epoch, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((num_epoch, num_instances, num_features)) #Initialize theta_hist
    loss_hist = np.zeros((num_epoch, num_instances)) #Initialize loss_hist
    #TODO


def load_data():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('ridge_regression_dataset.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    print('Split into Train and Test')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)

    print("Scaling all to [0, 1]")
    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    return X_train, y_train, X_test, y_test


## running stuff. 
X_train, y_train, X_test, y_test= load_data()
##question 12 
for i in [.05,.025, .01]:
    a,b=batch_grad_descent(X_train, y_train, alpha=i, grad_check=True)
    #print("hi")
    #print(b)
    t = np.arange(1001)
    plt.plot(t,b,label="alpha={0}".format(i))
    plt.xlabel("iteration number")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylabel("average suare loss")
    plt.grid() 
    plt.title("average training loss for difrent learning rates (alpha)")
    plt.legend()
plt.show()

## question 13
for i in [.05,.025, .01]:
    a,b=test_loss_for_batch_grad_descent(X_train, y_train,X_test,y_test, alpha=i,grad_check=True)
    t = np.arange(1001)
    plt.plot(t,b,label="alpha={0}".format(i))
    plt.xlabel("iteration number")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.ylabel("average suare loss")
    plt.grid() 
    plt.legend()
plt.title("average test loss for difrent learning rates (alpha)")
plt.show()




## question 17

for i in [math.pow(10,-5),math.pow(10,-3),.1,1 ,10 ]:
    a,b=regularized_grad_descent(X_train, y_train, alpha=.05,lambda_reg=i)
    t = np.arange(1001)
    plt.plot(t,b,label="lambda={0}".format(i))
    plt.xlabel("iteration number")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("average suare loss")
    plt.grid() 
    plt.title("average training loss for difrent coefcients of regularization (lambda)")
    plt.legend()
plt.show()
for i in [math.pow(10,-7),math.pow(10,-5),math.pow(10,-3),.1,1 ,10 ]:
    a,b=test_loss_for_regularized_grad_descent(X_train, y_train,X_test, y_test, alpha=.05,lambda_reg=i)
    t = np.arange(1001)
    plt.plot(t,b,label="lambda={0}".format(i))
    plt.xlabel("iteration number")
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel("average suare loss")
    plt.grid() 
    plt.title("average test loss for difrent coefcients of regularization (lambda)")
    plt.legend()
plt.show()
# question 18
temp=[]
lambdas=[0,math.pow(10,-7),math.pow(10,-5),math.pow(10,-3)]
for i in lambdas:
    a,b=regularized_grad_descent(X_train, y_train, alpha=.05,lambda_reg=i)
    temp.append(b[-1])
plt.plot(lambdas,temp,label="lambda={0}".format(i))
plt.xlabel("iter")
plt.xscale('log')
#plt.yscale('log')
plt.ylabel("average suare loss")
plt.grid() 
plt.title("average test loss for difrent coefcients of regularization (lambda)")
plt.legend()
plt.show()

def classification_error(estimator,observations, lables):
    preds=estimator.predict(observations)
    return np.sum(preds!=lables)/len(lables)
