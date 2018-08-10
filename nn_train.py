## Copyright by strongnine. 


import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def relu(x):
    s = np.maximum(0, x)
    return s

def tanh(x):
    s = np.tanh(x)
    return s

def load_data():
    """
	create a series point labeled by 1 and 0 as the trainsets and testsets.
	"""
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

def initialize_parameters(layer_dims):
    """
	initialize the paramaters of each layer, 
	layer_dims: is a list that stores the size of each layer from the input layer to the output layer.
    """
    Weight = {}
    bias = {}
    for d in range(1, len(layer_dims)):
        Weight['W'+str(d)] = np.random.randn(layer_dims[d],layer_dims[d-1]) * 0.01
        bias['b'+str(d)] = np.zeros((layer_dims[d],1))

    return Weight, bias

def forward_propagation(X, Weight, bias, activation):
    """
	activation: Is a list of activation functions for each layer from the first hidden layer to the output layer
	""" 
    assert(len(Weight) == len(bias))
    assert(len(Weight) == len(activation))

    Z = {}
    A = {}
    A['A0'] = X
    L = len(Weight)
    for l in range(1,L + 1):
        Z['Z' + str(l)] = np.dot(Weight['W' + str(l)], A['A'+str(l-1)]) + bias['b'+str(l)]
        exec("A['A' + str(l)] = " + activation[l-1] + "(Z['Z' + str(l)])")

    return A, Z

def compute_cost(A, Y, Weight, lambd):
    m = Y.shape[1]
    L2_term = 0
    for key in Weight.keys():
        L2_term += (np.sum(np.square(Weight[key])))

    logprobs = np.multiply(-np.log(A['A'+str(len(A)-1)]) ,Y) + np.multiply(-np.log(1 - A['A'+str(len(A)-1)]), 1 - Y)
    cost = 1./ m * np.nansum(logprobs)
    cost += L2_term * lambd / (2 * m)
    return cost

def backward_propagation(X, Y, Weight, bias, A, activation):
    m = X.shape[1]
    gradients = {}
    L = len(Weight)
    gradients['dZ' + str(L)] = A['A' + str(L)] - Y
    gradients['dW' + str(L)] = 1./m * np.dot(gradients['dZ' + str(L)], A['A' + str(L-1)].T)
    gradients['db' + str(L)] = 1./m * np.sum(gradients['dZ' + str(L)], axis=1, keepdims = True)
    for l in range(L-1, 0, -1):
        gradients['dA' + str(l)] = np.dot(Weight['W'+str(l+1)].T, gradients['dZ'+str(l+1)])
        if activation[l-1] == 'relu':
            gradients['dZ'+str(l)] = np.multiply(gradients['dA'+str(l)], np.int64(A['A'+str(l)] > 0))
        elif activation[l-1] == 'tanh':
            gradients['dZ'+str(l)] = np.multiply(gradients['dA'+str(l)], 1 - np.power(A['A'+str(1)], 2))


        gradients['dW'+str(l)] = 1./m * np.dot(gradients['dZ'+str(l)], A['A'+str(l-1)].T)
        gradients['db'+str(l)] = 1./m * np.sum(gradients['dZ'+str(l)], axis=1, keepdims=True)

    return gradients

def updata_parameters(Weight, bias, gradients, lr = 0.1):
    for i in range(1, len(Weight)+1):
        Weight['W'+str(i)] -= lr * gradients['dW'+str(i)]
        bias['b'+str(i)] -= lr * gradients['db'+str(i)]

    return Weight, bias


def cls_predict(X, Weight, bias, activation):
    A, Z = forward_propagation(X, Weight, bias, activation)
    prediction = (A['A'+str(len(A)-1)] > 0.5)

    return prediction

def nn_train(X,Y, Weight, bias, activation, lr=0.1, lambd=0.7, num_iterations=5000, print_cost=[True, 1000]):
    for i in range(num_iterations):
        A, Z = forward_propagation(X, Weight, bias, activation)
        cost = compute_cost(A, Y, Weight, lambd)
        grads = backward_propagation(X, Y, Weight, bias, A, activation)
        Weight, b  = updata_parameters(Weight, bias, grads, lr)

        if print_cost[0] and i % print_cost[1] == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return Weight, bias

def plot(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()


train_X, train_Y, test_X, test_Y = load_data()
## set the layer dims and activation for each layer.
## you can change the list to build your neural network.
layer_dims = [2,5,20,1]
activation = ['relu', 'relu', 'sigmoid']
Weight, bias = initialize_parameters(layer_dims)

nn_train(train_X, train_Y, Weight, bias, activation, lr=0.1, lambd=0.7, num_iterations=30000)
predictions = cls_predict(test_X, Weight, bias, activation)
accuracy = np.mean((predictions[0,:] == test_Y[0,:]))
print(accuracy)

## Visualize the training results
axes = plt.gca()
axes.set_xlim([-1.20,1.20])
axes.set_ylim([-1.20,1.20])
plot(lambda x: cls_predict(x.T, Weight, bias, activation), test_X, test_Y)