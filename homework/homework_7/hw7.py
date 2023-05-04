"""Computation graph node types

Nodes must implement the following methods:
__init__   - initialize node
forward    - (step 1 of backprop) retrieve output ("out") of predecessor nodes (if
             applicable), update own output ("out"), and set gradient ("d_out") to zero
backward   - (step 2 of backprop), assumes that forward pass has run before.
             Also assumes that backward has been called on all of the node's
             successor nodes, so that self.d_out contains the
             gradient of the graph output with respect to the node output.
             Backward computes summands of the derivative of graph output with
             respect to the inputs of the node, corresponding to paths through the graph
             that go from the node's input through the node to the graph's output.
             These summands are added to the input node's d_out array.
get_predecessors - return a list of the node's parents

Nodes must furthermore have a the following attributes:
node_name  - node's name (a string)
out      - node's output
d_out    - derivative of graph output w.r.t. node output

This computation graph framework was designed and implemented by
Philipp Meerkamp, Pierre Garapon, and David Rosenberg.
License: Creative Commons Attribution 4.0 International License
"""

import numpy as np


class ValueNode(object):
    """Computation graph node having no input but simply holding a value"""
    def __init__(self, node_name):
        self.node_name = node_name
        self.out = None
        self.d_out = None

    def forward(self):
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        pass

    def get_predecessors(self):
        return []


class VectorScalarAffineNode(object):
    """ Node computing an affine function mapping a vector to a scalar."""
    def __init__(self, x, w, b, node_name):
        """ 
        Parameters:
        x: node for which x.out is a 1D numpy array
        w: node for which w.out is a 1D numpy array of same size as x.out
        b: node for which b.out is a numpy scalar (i.e. 0dim array)
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.x = x
        self.w = w
        self.b = b

    def forward(self):
        self.out = np.dot(self.x.out, self.w.out) + self.b.out
        self.d_out = np.zeros(self.out.shape)
        return self.out
##
    def backward(self):
        d_x = self.d_out * self.w.out
        d_w = self.d_out * self.x.out
        d_b = self.d_out
        self.x.d_out += d_x
        self.w.d_out += d_w
        self.b.d_out += d_b

    def get_predecessors(self):
        return [self.x, self.w, self.b]


class SquaredL2DistanceNode(object):
    """ Node computing L2 distance (sum of square differences) between 2 arrays."""
    def __init__(self, a, b, node_name):
        """ 
        Parameters:
        a: node for which a.out is a numpy array
        b: node for which b.out is a numpy array of same shape as a.out
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.a = a
        self.b = b
        # Variable for caching values between forward and backward
        self.a_minus_b = None

    def forward(self):
        self.a_minus_b = self.a.out - self.b.out
        self.out = np.sum(self.a_minus_b ** 2)
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        d_a = self.d_out * 2 * self.a_minus_b
        self.b.d_out=self.b.d_out.reshape(-1,1)
        d_b = -self.d_out * 2 * self.a_minus_b
        self.a.d_out += d_a
        self.b.d_out += d_b
        return self.d_out

    def get_predecessors(self):
        return [self.a, self.b]


class L2NormPenaltyNode(object):
    """ Node computing l2_reg * ||w||^2 for scalars l2_reg and vector w"""
    def __init__(self, l2_reg, w, node_name):
        """ 
        Parameters:
        l2_reg: a numpy scalar array (e.g. np.array(.01)) (not a node)
        w: a node for which w.out is a numpy vector
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.l2_reg = np.array(l2_reg)
        self.w = w

    def forward(self):
        self.out = self.l2_reg * self.w.out.T @ self.w.out
        self.d_out = np.zeros(self.out.shape)
        return self.out
    def backward(self):
        d_w = self.d_out * 2 * self.l2_reg * self.w.out
        self.w.d_out = d_w
        return self.d_out
    def get_predecessors(self):
        ## Your code
        return [self.w]


class SumNode(object):
    """ Node computing a + b, for numpy arrays a and b"""
    def __init__(self, a, b, node_name):
        """ 
        Parameters:
        a: node for which a.out is a numpy array
        b: node for which b.out is a numpy array of the same shape as a
        node_name: node's name (a string)
        """
        self.node_name = node_name
        self.out = None
        self.d_out = None
        self.b = b
        self.a = a

    def forward(self):
        self.out = self.a.out + self.b.out 
        self.d_out = np.zeros(self.out.shape)
        return self.out

    def backward(self):
        da = self.d_out 
        db = self.d_out 
        self.a.d_out += da
        self.b.d_out += db
        return self.d_out
    
    def get_predecessors(self):
        return [self.a, self.b]

class AffineNode(object):
    """Node implementing affine transformation (W,x,b)-->Wx+b, where W is a matrix,
    and x and b are vectors
        Parameters:
        W: node for which W.out is a numpy array of shape (m,d)
        x: node for which x.out is a numpy array of shape (d)
        b: node for which b.out is a numpy array of shape (m) (i.e. vector of length m)
    """
    def __init__(self, W, x, b, node_name):
        self.W = W
        self.x = x
        self.b = b
        self.node_name = node_name
    def forward(self):
        #self.out = self.W.out @ self.x.out + self.b.out ## affine transformation
        self.out = np.dot(self.W.out, self.x.out) + self.b.out ## affine transformation
        
        # if(self.out.shape == ()):
        #     self.d_out = np.zeros((1))    
        # else:
        self.d_out = np.zeros((self.out.shape))

        return self.out
    def backward(self):
        dW = self.d_out.reshape(-1, 1) @ self.x.out.reshape(1, -1)
        db = self.d_out
        # if(self.W.node_name == "w2"):
        #     self.W.out = self.W.out.reshape(1, -1)
        #     dW = dW.flatten()
        #     db = db.flatten()[0]
        dx = self.W.out.T @ self.d_out 
        self.W.d_out = dW
        self.x.d_out = dx
        self.b.d_out = db
        return self.d_out
    def get_predecessors(self):
       return [self.W, self.x, self.b]
        


class TanhNode(object):
    """Node tanh(a), where tanh is applied elementwise to the array a
        Parameters:
        a: node for which a.out is a numpy array
    """
    def __init__(self, a , node_name):
        self.a = a
        self.node_name = node_name
    def forward(self):
        self.out = np.tanh(self.a.out)
        self.d_out = np.zeros(self.out.shape)
        return self.out
    def backward(self):
        da = 0
        ###print("size of tanhout", self.out.shape)
        
        self.a.d_out += self.d_out * (1 - np.square(self.out))
        return self.d_out
    
    def get_predecessors(self):
        return [self.a]
        


class SoftmaxNode(object):
    """ Softmax node
        Parameters:
        z: node for which z.out is a numpy array
    """
    def __init__(self, z, node_name):
        self.z = z
        self.node_name = node_name
    def forward(self):
        exp_array = np.exp(self.z.out)
        self.out = exp_array / sum(exp_array)
        ###print(self.z.node_name)
        self.d_out = np.zeros( ( self.out.shape[0]) )
        return self.out
    def backward(self):
        out_vector = self.out.reshape((-1,1))
        dz = np.diagflat(self.out) - np.dot(out_vector, out_vector.T)
        #print("dz is", dz.shape)
        #print(self.d_out.shape)
        d_z_prime = self.d_out @ dz.T
        ###print(d_z_prime.shape, "prime")
        ###print("z_shape", self.z.d_out.shape )
        self.z.d_out += d_z_prime
        ###print(self.z.d_out.shape, "zout")
        return self.d_out
    def get_predecessors(self):
        return [self.z]


class NLLNode(object):
    """ Node computing NLL loss between 2 arrays.
        Parameters:
        y_hat: a node that contains all predictions
        y_true: a node that contains all labels
    """
    def __init__(self, y_hat, y_true, node_name) -> None:
        self.y_hat = y_hat
        self.y_true = y_true
        self.node_name = node_name
    def forward(self):
        self.out = np.argmax(np.sum(np.log(self.y_hat.out))) 
        self.d_out = np.zeros((self.out.shape))
        return self.out
    def backward(self):
        d_y_hat = -(self.out * (self.out -1))
        #d_y_hat = self.out - self.y_true.out
        ##print("here", d_y_hat)
        #print("this is ", d_y_hat)
        #print("there ", self.d_out)
        self.y_hat.d_out += self.d_out * d_y_hat  
        #print( self.y_hat.d_out)
        ###print("went thru")
        return self.d_out
    def get_predecessors(self):
        return [self.y_hat, self.y_true] ## maybee add back in y_true


## Ridge.pyç≈
import setup_problem
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import nodes
import graph
import plot_utils

class RidgeRegression(BaseEstimator, RegressorMixin):
    """ Ridge regression with computation graph """
    def __init__(self, l2_reg=1, step_size=.005,  max_num_epochs = 5000):
        self.max_num_epochs = max_num_epochs
        self.step_size = step_size

        # Build computation graph
        self.x = nodes.ValueNode(node_name="x") # to hold a vector input
        self.y = nodes.ValueNode(node_name="y") # to hold a scalar response
        self.w = nodes.ValueNode(node_name="w") # to hold the parameter vector
        self.b = nodes.ValueNode(node_name="b") # to hold the bias parameter (scalar)
        self.prediction = nodes.VectorScalarAffineNode(x=self.x, w=self.w, b=self.b,
                                                 node_name="prediction")
        self.objective = nodes.SumNode(a= nodes.SquaredL2DistanceNode(a=self.prediction, b=self.y,node_name="square loss"), b= nodes.L2NormPenaltyNode(l2_reg, self.w , node_name="l2 reg"), node_name="ridge objective")
        self.inputs = [self.x]
        self.outcomes = [self.y]
        self.parameters = [self.w, self.b ]
        self.graph = graph.ComputationGraphFunction(self.inputs, self.outcomes,
                                                          self.parameters, self.prediction,
                                                          self.objective)
        # TODO: ADD YOUR CODE HERE

        
    def fit(self, X, y):
        num_instances, num_ftrs = X.shape
        y = y.reshape(-1)

        init_parameter_values = {"w": np.zeros(num_ftrs), "b": np.array(0.0)}
        self.graph.set_parameters(init_parameter_values)

        for epoch in range(self.max_num_epochs):
            shuffle = np.random.permutation(num_instances)
            epoch_obj_tot = 0.0
            for j in shuffle:
                obj, grads = self.graph.get_gradients(input_values = {"x": X[j]},
                                                    outcome_values = {"y": y[j]})
                ###pr(obj)
                epoch_obj_tot += obj
                # Take step in negative gradient direction
                steps = {}
                for param_name in grads:
                    steps[param_name] = -self.step_size * grads[param_name]
                self.graph.increment_parameters(steps)

            if epoch % 50 == 0:
                train_loss = sum((y - self.predict(X,y)) **2)/num_instances
                ##pr("Epoch ",epoch,": Ave objective=",epoch_obj_tot/num_instances," Ave training loss: ",train_loss)

    def predict(self, X, y=None):
        try:
            getattr(self, "graph")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        num_instances = X.shape[0]
        preds = np.zeros(num_instances)
        for j in range(num_instances):
            preds[j] = self.graph.get_prediction(input_values={"x":X[j]})

        return preds



def main():
    data_fname = "data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = setup_problem.load_problem(data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    pred_fns = []
    x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))
    X = featurize(x)

    l2reg = 1
    estimator = RidgeRegression(l2_reg=l2reg, step_size=0.00005, max_num_epochs=2000)
    estimator.fit(X_train, y_train)
    name = "Ridge with L2Reg="+str(l2reg)
    pred_fns.append({"name":name, "preds": estimator.predict(X) })


    l2reg = 0
    estimator = RidgeRegression(l2_reg=l2reg, step_size=0.0005, max_num_epochs=500)
    estimator.fit(X_train, y_train)
    name = "Ridge with L2Reg="+str(l2reg)
    pred_fns.append({"name":name, "preds": estimator.predict(X) })

    # Let's plot prediction functions and compare coefficients for several fits
    # and the target function.

    pred_fns.append({"name": "Target Parameter Values (i.e. Bayes Optimal)", "coefs": coefs_true, "preds": target_fn(x)})

    plot_utils.plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")

if __name__ == '__main__':
  main()


## mlp.py
import matplotlib.pyplot as plt
import setup_problem
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import nodes
import graph
import plot_utils
import pdb
#pdb.set_trace()


class MLPRegression(BaseEstimator, RegressorMixin):
    """ MLP regression with computation graph """
    def __init__(self, num_hidden_units=10, step_size=.005, init_param_scale=0.01, max_num_epochs = 5000):
        self.num_hidden_units = num_hidden_units
        self.init_param_scale = init_param_scale
        self.max_num_epochs = max_num_epochs
        self.step_size = step_size

        # # Build computation graph
        # # TODO: ADD YOUR CODE HERE
        # """ 
        # Parameters:
        # inputs: list of ValueNode objects containing inputs (in the ML sense)
        # outcomes: list of ValueNode objects containing outcomes (in the ML sense)
        # parameters: list of ValueNode objects containing values we will optimize over
        # prediction: node whose 'out' variable contains our prediction
        # objective:  node containing the objective for which we compute the gradient
        # """
        self.W = [nodes.ValueNode(node_name = "W1"),nodes.ValueNode(node_name = "w2")]
        self.x = nodes.ValueNode("x")
        self.b = [nodes.ValueNode(node_name = "b1"),nodes.ValueNode(node_name = "b2")]
        self.y = nodes.ValueNode("y")
        
        self.a = nodes.AffineNode(self.W[0], self.x, self.b[0], "affine")
        self.h = nodes.TanhNode(a = self.a,node_name="tanh")
        self.inputs = [self.x]
        self.outcomes = [self.y]
        self.parameters = self.W + self.b 
        self.prediction = nodes.VectorScalarAffineNode(w=self.W[1],x=self.h, b=self.b[1],node_name= "prediction")
        self.objective = nodes.SquaredL2DistanceNode(a=self.prediction, b=self.y, node_name="objective")
        self.graph = graph.ComputationGraphFunction(inputs=self.inputs, outcomes= self.outcomes, 
                                        parameters=self.parameters, prediction=self.prediction, objective=self.objective )
    def fit(self, X, y):
        num_instances, num_ftrs = X.shape
        y = y.reshape(-1)
        s = self.init_param_scale
        init_values = {"W1": s * np.random.standard_normal((self.num_hidden_units, num_ftrs)),
                       "b1": s * np.random.standard_normal((self.num_hidden_units)),
                       "w2": s * np.random.standard_normal((self.num_hidden_units)),
                       "b2": s * np.array(np.random.randn()) }

        self.graph.set_parameters(init_values)

        for epoch in range(self.max_num_epochs):
            shuffle = np.random.permutation(num_instances)
            epoch_obj_tot = 0.0
            for j in shuffle:
                obj, grads = self.graph.get_gradients(input_values = {"x": X[j]},
                                                    outcome_values = {"y": y[j]})
                ###pr(obj)
                epoch_obj_tot += obj
                # Take step in negative gradient direction
                steps = {}
                for param_name in grads:
                    steps[param_name] = -self.step_size * grads[param_name]
                self.graph.increment_parameters(steps)
                #pdb.set_trace()

            if epoch % 50 == 0:
                train_loss = sum((y - self.predict(X,y)) **2)/num_instances
                ##pr("Epoch ",epoch,": Ave objective=",epoch_obj_tot/num_instances," Ave training loss: ",train_loss)

    def predict(self, X, y=None):
        try:
            getattr(self, "graph")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        num_instances = X.shape[0]
        preds = np.zeros(num_instances)
        for j in range(num_instances):
            preds[j] = self.graph.get_prediction(input_values={"x":X[j]})

        return preds



def main():
    data_fname = "data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = setup_problem.load_problem(data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    # Let's plot prediction functions and compare coefficients for several fits
    # and the target function.
    pred_fns = []
    x = np.sort(np.concatenate([np.arange(0,1,.001), x_train]))

    pred_fns.append({"name": "Target Parameter Values (i.e. Bayes Optimal)", "coefs": coefs_true, "preds": target_fn(x)})

    estimator = MLPRegression(num_hidden_units=10, step_size=0.001, init_param_scale=.0005,  max_num_epochs=5000)
    x_train_as_column_vector = x_train.reshape(x_train.shape[0],1) # fit expects a 2-dim array
    x_as_column_vector = x.reshape(x.shape[0],1) # fit expects a 2-dim array
    estimator.fit(x_train_as_column_vector, y_train)
    name = "MLP regression - no features"
    pred_fns.append({"name":name, "preds": estimator.predict(x_as_column_vector) })
    #plot_utils.plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")
    if (1==1):
        X = featurize(x)
        estimator = MLPRegression(num_hidden_units=10, step_size=0.0005, init_param_scale=.01,  max_num_epochs=500)
        estimator.fit(X_train, y_train)
        name = "MLP regression - with features"
        pred_fns.append({"name":name, "preds": estimator.predict(X) })
        plot_utils.plot_prediction_functions(x, pred_fns, x_train, y_train, legend_loc="best")

if __name__ == '__main__':
  main()
## multiclass.py
import setup_problem
from sklearn.base import BaseEstimator, RegressorMixin
try:
    from sklearn.datasets.samples_generator import make_blobs
except:
    from sklearn.datasets import make_blobs
import numpy as np
import nodes
import graph

def calculate_nll(y_preds, y):
    """
    Function that calculate the average NLL loss
    :param y_preds: N * C probability array
    :param y: N int array
    :return:
    """
    return np.mean(-np.log(y_preds)[np.arange(len(y)),y])


class MulticlassClassifier(BaseEstimator, RegressorMixin):
    """ Multiclass prediction """
    def __init__(self, num_hidden_units=10, step_size=.005, init_param_scale=0.01, max_num_epochs = 1000, num_class=3):
        self.num_hidden_units = num_hidden_units
        self.init_param_scale = init_param_scale
        self.max_num_epochs = max_num_epochs
        self.step_size = step_size
        self.num_class = num_class

        # Build computation graph
        # TODO: add your code here
        self.W = [nodes.ValueNode(node_name = "W1"),nodes.ValueNode(node_name = "W2")]
        self.x = nodes.ValueNode("x")
        self.b = [nodes.ValueNode(node_name = "b1"),nodes.ValueNode(node_name = "b2")]
        self.y = nodes.ValueNode("y")
        self.a = nodes.AffineNode(self.W[0], self.x, self.b[0], "affine")
        self.h = nodes.TanhNode(a = self.a,node_name="tanh")
        self.inputs = [self.x]
        self.outcomes = [self.y]
        self.parameters = self.W + self.b 
        self.z = nodes.AffineNode(W=self.W[1],x=self.h, b=self.b[1],node_name= "z")
        self.prediction = nodes.SoftmaxNode(z=self.z, node_name="soft max")
        self.objective = nodes.NLLNode(y_hat=self.prediction, y_true=self.y, node_name="objective")
        self.graph = graph.ComputationGraphFunction(inputs=self.inputs, outcomes= self.outcomes, 
                                        parameters=self.parameters, prediction=self.prediction, objective=self.objective )
    def fit(self, X, y):
        num_instances, num_ftrs = X.shape
        ##pr("intended shape of x is", X.shape)
        y = y.reshape(-1)
        s = self.init_param_scale
        init_values = {"W1": s * np.random.standard_normal((self.num_hidden_units, num_ftrs)),
                       "b1": s * np.random.standard_normal((self.num_hidden_units)),
                       "W2": np.random.standard_normal((self.num_class, self.num_hidden_units)),
                       "b2": np.array(np.random.randn(self.num_class)) }
        self.graph.set_parameters(init_values)

        for epoch in range(self.max_num_epochs):
            shuffle = np.random.permutation(num_instances)
            epoch_obj_tot = 0.0
            for j in shuffle:
                obj, grads = self.graph.get_gradients(input_values = {"x": X[j]},
                                                    outcome_values = {"y": y[j]})
                ###pr(obj)
                epoch_obj_tot += obj
                # Take step in negative gradient direction
                steps = {}
                for param_name in grads:
                    steps[param_name] = -self.step_size * grads[param_name]
                self.graph.increment_parameters(steps)
                #pdb.set_trace()

            if epoch % 50 == 0:
                train_loss = calculate_nll(self.predict(X,y), y)
                ##pr("Epoch ",epoch," Ave training loss: ",train_loss)

    def predict(self, X, y=None):
        try:
            getattr(self, "graph")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")

        num_instances = X.shape[0]
        preds = []
        for j in range(num_instances):
            preds.append(self.graph.get_prediction(input_values={"x":X[j]}).reshape(1,-1))

        return np.concatenate(preds, axis=0)



def main():
    # load the data from HW5
    np.random.seed(2)
    X, y = make_blobs(n_samples=500, cluster_std=.25, centers=np.array([(-3, 1), (0, 2), (3, 1)]))
    training_X = X[:300]
    training_y = y[:300]
    test_X = X[300:]
    test_y = y[300:]

    # train the model
    estimator = MulticlassClassifier()
    estimator.fit(training_X, training_y)

    # report test accuracy
    test_acc = np.sum(np.argmax(estimator.predict(test_X), axis=1)==test_y)/len(test_y)
    ##pr("Test set accuracy = {:.3f}".format(test_acc))


if __name__ == '__main__':
  main()