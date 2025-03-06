#لایه های شبکه عصبی
import numpy as NP
from neuron import Neuron
class Layers:
    def __init__(self,num_neurons,input_size):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]
        self.output = None 
    def feedforward(self,inputs):
        self.inputs = inputs
        self.output = NP.array([neuron.feedforward(inputs) for neuron in self.neurons]).T
        return self.output
    def backward(self, error, learning_rate):
        """ اجرای پس‌انتشار خطا و به‌روزرسانی وزن‌ها و بایاس‌ها """
        sigmoid_derivative = self.output * (1 - self.output)  # مشتق سیگموید
        
        # محاسبه دلتا برای این لایه
        delta = error * sigmoid_derivative  

        # خطای لایه قبل را محاسبه می‌کنیم
        prev_error = NP.dot(delta, NP.array([neuron.weight for neuron in self.neurons]))

        # به‌روزرسانی وزن‌ها و بایاس‌ها
        for i, neuron in enumerate(self.neurons):
            neuron.weight += learning_rate * NP.dot(self.inputs.T, delta[:, i])  # آپدیت وزن‌ها
            neuron.bias += learning_rate * NP.sum(delta[:, i])  # آپدیت بایاس‌ها

        return prev_error  # برگرداندن خطا برای لایه قبلی