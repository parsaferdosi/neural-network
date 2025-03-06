#لایه های شبکه عصبی
import numpy as NP
from neuron import Neuron
class Layers:
    def __init__(self,num_neurons,input_size):
        self.neurons=[]
        for i in range(num_neurons):
            weight=NP.random.rand(input_size,1)
            bias=NP.random.rand(1)
            self.neurons.append(Neuron(weight,bias))
    def feedforward(self,inputs):
        outputs=[]
        for neuron in self.neurons:
            outputs.append(neuron.feedforward(inputs))
        return NP.array(outputs).T