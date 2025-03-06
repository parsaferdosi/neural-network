import numpy as NP
from layers import Layers
class NeuralNetwork:
    def __init__(self , layer_size):
        self.layers=[]
        for i in range(len((layer_size))-1):
            self.layers.append(Layers(num_neurons=layer_size[i+1],input_size=layer_size[i]))
    def feedforward(self,inputs):
        for layer in self.layers:
            inputs=layer.feedforward(inputs)
        return inputs
layer_sizes = [4, 3, 2]  # اندازه ورودی ۴، لایه پنهان با ۳ نورون، و خروجی با ۲ نورون
network = NeuralNetwork(layer_sizes)
inputs = NP.array([0.5, -0.6, 0.1, 0.2])
output = network.feedforward(inputs)
print(output)