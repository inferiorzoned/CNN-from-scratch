# implement cnn from scratch
"""
In this assignment, you will have to implement a convolutional neural network for an image
classification task. There will be six basic components in your neural network:
    1. Convolution layer: there will be four (hyper)parameters: the number of output channels,
    filter dimension, stride, padding.
    2. Activation layer: implement an element-wise ReLU.
    3. Max-pooling layer: there will be two parameters: filter dimension, stride.
    4. Fully-connected layer: a dense layer. There will be one parameter: output dimension.
    5. Flattening layer: it will convert a (series of) convolutional filter maps to a column vector.
    6. Softmax layer: it will convert final layer projections to normalized probabilities
The model architecture will be given in a text file. A sample architecture is shown for your
convenience.
Conv 6 5 1 2
ReLU
Pool 2 2
Conv 12 5 1 0
ReLU
Pool 2 2
Conv 100 5 1 0
ReLU
FC 10
Softmax

You will have to implement the backpropagation algorithm to train the model. The weights will be
updated using batch gradient descent, where instead of optimizing with the loss calculated over
all training samples, you will update gradients with a subset of the training set (ideally 32
samples) in each step.

You will work with two datasets: MNIST and CIFAR-10. Both datasets are openly available and
have 50k-60k samples. Split the evaluation set into half so that you can use 5k samples for
validation and 5k samples for test purposes. You will also be given a toy dataset to test whether
or not your implementation of the backpropagation algorithm works correctly
"""

import numpy as np
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical


from termcolor import colored
def print_g(input_list):
    print(colored(input_list, "green"))
def print_y(input_list):
    print(colored(input_list, "yellow"))
def print_m(input_list):
    print(colored(input_list, "magenta"))


def nparray_head_or_tail(x: np.array, n:int, head_or_tail="head") ->np.array:
    """
    Returns head or tail N elements of array.
    :param x: Numpy array.
    :param n: N elements to return on end or start.
    :return: Last N elements of array.
    """
    if n == 0:
        return x[0:0]  # Corner case: x[-0:] will return the entire array but head_or_tail(0) should return an empty array.
    elif head_or_tail == "tail":
        return x[-n:]  # Normal case: last N elements of array.
    elif head_or_tail == "head":
        return x[0:n]  # Normal case: first N elements of array.

def normalize_data(X_train, X_test, y_train, y_test):
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    assert X_train.shape == (60000, 28, 28)
    assert X_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,)

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train, X_test, y_train, y_test = normalize_data(X_train, X_test, y_train, y_test)

    X_valid = nparray_head_or_tail(X_test, 5000, "head")
    X_test = nparray_head_or_tail(X_test, 5000, "tail")

    y_valid = nparray_head_or_tail(y_test, 5000, "head")
    y_test = nparray_head_or_tail(y_test, 5000, "tail")

    print(nparray_head_or_tail(y_test, 5, "tail"))

    return X_train, y_train, X_valid, y_valid, X_test, y_test

class ConvLayer:
    def __init__(self, num_channels, filter_dim, stride, padding):
        self.num_channels = num_channels
        self.filter_dim = filter_dim
        self.stride = stride
        self.padding = padding

    def pad_input(self, X):
        """
        Pads the input with zeros.
        :param X: Input array, shape = (batch_size, height, width, num_filters)
        :return: Padded input array, shape = (batch_size, height+2p, width+2p, num_filters)
        """
        if self.padding == 0:
            return X
        pad_dim = self.padding
        pad_width = ((0, 0), (pad_dim, pad_dim), (pad_dim, pad_dim), (0, 0))
        return np.pad(X, pad_width, 'constant', constant_values=0)

    def convolve_single_pass(self, input_section, W):
        """
        Performs a single pass of convolution.
        :param input_section: Input section, shape = (filter_dim, filter_dim, num_channels)
        :param W: Filter weights, shape = (filter_dim, filter_dim, num_channels)
        :return: Output of convolution, shape = (filter_dim, filter_dim, num_channels)
        """ 
        return np.sum(input_section * W, axis=None)
    
    def convolve(self, X, W, b):
        """
        Performs convolution using self.stride
        :param X: Input array, shape = (batch_size, height, width, num_channels)
        :param W: Filter weights, shape = (filter_dim, filter_dim, num_channels)
        :param b: Filter bias, shape = (num_channels,)
        :return: Output of convolution, shape = (batch_size, height, width, num_channels)
        """
        X_padded = self.pad_input(X)
        output = np.zeros(X_padded.shape)
        for i in range(0, X_padded.shape[1] - self.filter_dim + 1, self.stride):
            for j in range(0, X_padded.shape[2] - self.filter_dim + 1, self.stride):
                input_section = X_padded[:, i:i + self.filter_dim, j:j + self.filter_dim, :]
                # temp = self.convolve_single_pass(input_section, W)
                # # print(temp)
                output[:, i:i + self.filter_dim, j:j + self.filter_dim, :] += self.convolve_single_pass(input_section, W) + b
        return output
        

class ReluLayer:
    def __init__(self):
        pass

    def forward(self, X):
        """
        Performs forward pass.
        :param X: Input array, shape = (batch_size, height, width, num_channels)
        :return: Output of forward pass, shape = (batch_size, height, width, num_channels)
        """
        return np.maximum(X, 0)

    def backward(self, X, dX):
        """
        Performs backward pass.
        :param X: Input array, shape = (batch_size, height, width, num_channels)
        :param dX: Gradient of loss with respect to output of forward pass, shape = (batch_size, height, width, num_channels)
        :return: Gradient of loss with respect to input of forward pass, shape = (batch_size, height, width, num_channels)
        """
        return dX * (X > 0)



class MaxPoolingLayer:
    def __init__(self, filter_dim, stride):
        self.filter_dim = filter_dim
        self.stride = stride

    def max_pool(self, X):
        """
        Performs max pooling using self.stride
        :param X: Input array, shape = (batch_size, height, width, num_filters)
        :return: Output of max pooling, shape = (batch_size, floor((height-filter_dim)/stride)+1, floor((weight-filter_dim)/stride)+1, num_filters)
        """
        output_height = int(np.floor((X.shape[1] - self.filter_dim) / self.stride)) + 1
        output_width = int(np.floor((X.shape[2] - self.filter_dim) / self.stride)) + 1
        output = np.zeros((X.shape[0], output_height, output_width, X.shape[3]))

        for i in range(0, X.shape[1] - self.filter_dim + 1, self.stride):
            for j in range(0, X.shape[2] - self.filter_dim + 1, self.stride):
                output[:, int(np.floor(i / self.stride)), int(np.floor(j / self.stride)), :] = np.max(
                    X[:, i:i + self.filter_dim, j:j + self.filter_dim, :], axis=None)
        return output

class FullyConnectedLayer:
    def __init__(self, output_dim):
        self.output_dim = output_dim

    # forward propagation method of fully-connected layer, taking batch_size as input
    def forward(self, X, W, b):
        """
        :param X: flattened input (1, num_inputs)
        :param W: weights, shape = (num_inputs, num_outputs)
        :param b: bias, shape = (num_outputs,)
        :return: output data, shape = (batch_size, self.output_dim)
        """
        return np.dot(X, W) + b

class SoftmaxLayer:
    def __init__(self):
        pass

    def forward(self, X):
        """
        :param X: flattened input (1, num_inputs)
        :return: output data, shape = (batch_size, self.output_dim)
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)
    
    def backward(self, X, dX):
        """
        :param X: flattened input (1, num_inputs)
        :param dX: gradient of loss with respect to output of forward pass
        :return: gradient of loss with respect to input of forward pass
        """
        return dX * (X / np.sum(X, axis=1, keepdims=True))


def flatten_array(X):
    """
    :param X: input data, shape = (ANY) --- total num_inputs data 
    :return: output data, shape = (1, num_inputs)
    """
    return X.reshape(1, -1)

def cross_entropy_loss(y, y_hat):
    """
    :param y: true label, shape = (1, num_classes)
    :param y_hat: predicted label, shape = (1, num_classes)
    :return: loss, shape = (1,)
    """
    # find where y is 1
    idx = np.where(y == 1)[1]
    print_m(y_hat[0, idx])
    return -np.log(y_hat[0, idx])
    # return -np.sum(y * np.log(y_hat))

def generate_batches(X, y, batch_size=32):
    """
    Generate a sequential list mini_batch of size 32
    :param X: input data, shape = (num_samples, height, width, num_channels)
    :param y: labels, shape = (num_samples,)
    :param batch_size: size of mini_batch
    :return: a list of batches (batch_x, batch_y)
    """
    num_samples = X.shape[0]
    num_batches = num_samples // batch_size
    remainder = False
    if num_samples % batch_size != 0:
        num_batches += 1
        remainder = True

    batches = []
    start_index = 0
    end_index = 0
    for i in range(num_batches - 1):
        start_index = i * batch_size
        # end_index = min((i + 1) * batch_size, num_samples)
        batch_x = X[start_index:start_index + batch_size]
        batch_y = y[start_index:start_index + batch_size]
        batches.append((batch_x, batch_y))
    
    start_index += batch_size
    batch_x = X[start_index:]
    batch_y = y[start_index:]
    batches.append((batch_x, batch_y))

    return batches

def build_model():
    # build model by parsing input.txt
    """
    input.txt format:
    Conv 6 5 1 2
    ReLU
    Pool 2 2
    Conv 12 5 1 0
    ReLU
    Pool 2 2
    Conv 100 5 1 0
    ReLU
    FC 10
    Softmax
    """
    with open('input.txt', 'r') as f:
        lines = f.readlines()
        layers = []
        for line in lines:
            if line.startswith('Conv'):
                layer = ConvLayer(int(line.split(' ')[1]), int(line.split(' ')[2]), int(line.split(' ')[3]))
                layers.append(layer)
            elif line.startswith('ReLU'):
                layer = ReluLayer()
                layers.append(layer)
            elif line.startswith('Pool'):
                layer = MaxPoolingLayer(int(line.split(' ')[1]), int(line.split(' ')[2]))
                layers.append(layer)
            elif line.startswith('FC'):
                layer = FullyConnectedLayer(int(line.split(' ')[1]))
                layers.append(layer)
            elif line.startswith('Softmax'):
                layer = SoftmaxLayer()
                layers.append(layer)
            else:
                print('Unknown layer type')
                return
    



# X_train, y_train, X_valid, y_valid, X_test, y_test = load_mnist_data()
# batches = generate_batches(X_train, y_train)
# print(X_train.shape[0])
# print(len(batches))

# test ConvLayer pad_input function
X = np.random.randn(2, 3, 3, 1)
conv_layer = ConvLayer(1, 3, 1, 1)
# print_m(X.shape)
# print_g(conv_layer.pad_input(X).shape)

# test ConvLayer convolve function
X = np.random.randn(2, 3, 3, 1)
W = np.random.randn(3, 3, 1)
conv_layer = ConvLayer(1, 3, 1, 1)
# print_m(X)
# print_g(W)
# print_m(conv_layer.convolve(X, W, 0))

# test PoolingLayer max_pool function
X = np.random.randn(4, 28, 28, 2)
pool_layer = MaxPoolingLayer(2, 2)
# print_m(X)
# print_g(pool_layer.max_pool(X).shape)

# test FullyConnectedLayer forward function after flattening
X = np.random.randn(2, 3)
X = flatten_array(X)
print(X.shape)
W = np.random.randn(6, 4)
b = np.random.randn(4)
fc_layer = FullyConnectedLayer(4)
# print_m(X)
# print_g(W)
# print_g(b)
# print_g(fc_layer.forward(X, W, b))

# test Softmax function
X = np.random.randn(2, 3)
X = flatten_array(X)
# print(X.shape)
# print_g(Softmax(X))

# test cross_entropy_loss function
y = np.array([[0, 1, 0]])
y_hat = np.array([[0.1, 0.9, 0.1]])
print_g(cross_entropy_loss(y, y_hat))








