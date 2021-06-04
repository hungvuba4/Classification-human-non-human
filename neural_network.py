import numpy as np
from layer import Layer


class NeuralNetwork:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.layers = []

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise Exception('Invalid type', type(layer), " != <class Layer>")
        self.layers.append(layer)

    def initialize(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.initialize(self.X)
            else:
                layer.initialize(self.layers[i - 1])
        output_layer = Layer(self.y.shape[1], act_func='sigmoid')
        output_layer.initialize(self.layers[len(self.layers) - 1])
        self.add_layer(output_layer)

    def feedforward(self):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(self.X)
            else:
                layer.forward(self.layers[i - 1])

    def backpropagation(self):
        delta = -self.y + self.layers[len(self.layers) - 1].a
        for i in range(len(self.layers) - 1, -1, -1):
            if i == 0:
                delta = self.layers[i].backward(delta, self.X, self.lr)
            else:
                delta = self.layers[i].backward(delta, self.layers[i - 1], self.lr)

    def loss_func(self, y_pred):
        return 1 / self.y.shape[0] * - np.sum(self.y * np.log(y_pred) - (1 - self.y) * np.log(1 - y_pred))

    def fit(self, lr, it):
        self.initialize()
        self.loss = []
        self.lr = lr
        self.it = it
        for i in range(it):
            self.feedforward()
            self.backpropagation()
            y_pred = self.layers[len(self.layers) - 1].a
            print(f'Epoch {i + 1}: {self.loss_func(y_pred)}')
            if i % 100 == 0:
                self.loss.append(self.loss_func(y_pred))

    # regression problem
    def is_continues(self):
        return len(np.unique(self.y)) > (self.y.shape[0] / 3)

    # multiclassification problem
    def is_multiclass(self):
        return len(np.unique(self.y)) > 2

    def thre(self, output_layer, thre):
        pred = output_layer.a
        pred[pred < thre] = 0
        pred[pred >= thre] = 1
        return pred

    def thre_multiclass(self, output_layer):
        pred = output_layer.a
        pred = np.where(pred == np.max(pred, keepdims=True, axis=1), 1, 0)
        return pred

    def pred(self, X_test):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.forward(X_test)
            else:
                layer.forward(self.layers[i - 1])
        if self.is_continues():
            return self.layers[len(self.layers) - 1].a
        if self.is_multiclass():
            return self.thre_multiclass(self.layers[len(self.layers) - 1])
        return self.thre(self.layers[len(self.layers) - 1], 0.5)
            
