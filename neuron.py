import numpy as np

class Neuron:
    def __init__(self, input_size, activation="relu"):
        self.weight = np.random.randn(input_size) * 0.01
        self.bias = np.zeros(1)
        self.activation = activation

    def activate(self, x):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)

    def activate_derivative(self, x):
        if self.activation == "sigmoid":
            return x * (1 - x)  # مشتق سیگموید
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)  # مشتق ReLU

    def feedforward(self, inputs):
        self.inputs = inputs
        self.output = self.activate(np.dot(inputs, self.weight) + self.bias)
        return self.output

    def backward(self, error, learning_rate):
        delta = error * self.activate_derivative(self.output)  
        self.weight += learning_rate * np.dot(self.inputs.T, delta)  
        self.bias += learning_rate * np.sum(delta, axis=0)
        return np.dot(delta, self.weight.T) 
