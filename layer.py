import numpy as np
from act_func import get_act

'''
    Layer: The hidden layer of neural network
    ...

    Attributes:
    ---------
    shape (int): number of units in the hidden layer
    activation_function (string): activation function in the hidden layer
'''


class Layer:
    def __init__(self, shape, act_func='sigmoid'):
        self.shape = (shape,)
        self.act_func, self.dact_func = get_act(act_func)

    def initialize(self, pre_layer):
        self.shape = (pre_layer.shape[0],) + self.shape
        self.w = np.random.randn(pre_layer.shape[1], self.shape[1]) / (pre_layer.shape[1] * self.shape[1])
        self.b = np.random.randn(1, self.shape[1]) / self.shape[1]
        self.a = np.zeros(self.shape)

    def forward(self, pre_layer):
        if isinstance(pre_layer, np.ndarray):
            self.z = np.dot(pre_layer, self.w) + self.b  # first hidden layer
        else:
            self.z = np.dot(pre_layer.a, self.w) + self.b
        self.a = self.act_func(self.z)

    def backward(self, delta, pre_layer, lr):
        delta = delta * self.dact_func(self.z)
        db = np.sum(delta)
        if isinstance(pre_layer, np.ndarray):
            dw = np.dot(pre_layer.T, delta)  # first hidden layer
        else:
            dw = np.dot(pre_layer.a.T, delta)
        self.b -= lr * db
        delta = np.dot(delta, self.w.T)
        self.w -= lr * dw
        return delta
