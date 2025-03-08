import numpy as np
from neuron import Neuron

class Layers:
    def __init__(self, num_neurons, input_size, activation="relu"):
        self.activation = activation
        self.neurons = [Neuron(input_size, activation) for _ in range(num_neurons)]
        self.output = None  
        self.raw_inputs = None  

    def softmax(self, x):
        """ اعمال Softmax به کل لایه به صورت ماتریسی """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # جلوگیری از overflow
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def softmax_derivative(self, softmax_output):
        """ محاسبه مشتق Softmax (بر اساس ماتریس ژاکوبین) """
        batch_size, num_classes = softmax_output.shape
        jacobian = np.zeros((batch_size, num_classes, num_classes))

        for i in range(batch_size):
            s = softmax_output[i].reshape(-1, 1)
            jacobian[i] = np.diagflat(s) - np.dot(s, s.T)

        return jacobian  

    def feedforward(self, inputs):
        self.inputs = inputs
        self.raw_inputs = np.dot(inputs, np.array([neuron.weight for neuron in self.neurons]).T) + \
                          np.array([neuron.bias for neuron in self.neurons]).T  

        if self.activation == "softmax":
            self.output = self.softmax(self.raw_inputs)  # استفاده از softmax برای کل لایه
        else:
            self.output = np.vstack([neuron.activate(self.raw_inputs[:, i]) for i, neuron in enumerate(self.neurons)]).T
        return self.output

    def backward(self, error, learning_rate):
        """ انتشار خطا به عقب """
        if self.activation == "softmax":
            softmax_derivative = self.softmax_derivative(self.output)
            error = np.einsum('bij,bj->bi', softmax_derivative, error)  # محاسبه Δ با ماتریس ژاکوبین

        activation_derivative = np.vstack([
            neuron.activate_derivative(self.raw_inputs[:, i]) for i, neuron in enumerate(self.neurons)
        ]).T
        delta = error * activation_derivative  

        batch_size = delta.shape[0]
        num_neurons = len(self.neurons)
        delta = delta.reshape(batch_size, num_neurons)
        weights = np.array([neuron.weight for neuron in self.neurons])  

        prev_error = np.dot(delta, weights)  
        self.inputs = self.inputs.reshape(batch_size, -1)  

        for i, neuron in enumerate(self.neurons):
            neuron.weight += learning_rate * np.dot(self.inputs.T, delta[:, i]) / batch_size  # نرمال‌سازی گرادیان‌ها
            neuron.bias += learning_rate * np.mean(delta[:, i], axis=0, keepdims=True)  
        return prev_error  
