import numpy as np
from neuron import Neuron

class Layers:
    def __init__(self, num_neurons, input_size, activation="relu"):
        self.neurons = [Neuron(input_size, activation) for _ in range(num_neurons)]
        self.output = None  

    def feedforward(self, inputs):
        self.inputs = inputs
        self.output = np.array([neuron.feedforward(inputs) for neuron in self.neurons]).T
        return self.output

    def backward(self, error, learning_rate):
        # محاسبه مشتق تابع فعال‌سازی (ReLU یا Sigmoid)
        activation_derivative = np.array([neuron.activate_derivative(neuron.output) for neuron in self.neurons]).T
        
        # محاسبه دلتا برای این لایه
        delta = error * activation_derivative  

        # ذخیره خطای لایه قبلی
        prev_error = np.dot(delta, np.array([neuron.weight for neuron in self.neurons]))  

        # اصلاح وزن و بایاس برای هر نورون
        for i, neuron in enumerate(self.neurons):
            neuron.weight += learning_rate * np.dot(self.inputs.T, delta[:, i])  # اصلاح وزن‌ها
            neuron.bias += learning_rate * np.sum(delta[:, i])  # اصلاح بایاس‌ها

        return prev_error  # برگرداندن خطای این لایه برای لایه قبلی
