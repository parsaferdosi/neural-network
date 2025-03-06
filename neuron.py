#کلاس نورون که پایه کاره
import numpy as NP
class Neuron:
    def __init__(self,input_size):
        self.weight=NP.random.randn(input_size)*0.1
        self.bias=NP.random.randn()*0.1
    def sigmoid(self,x):
        return 1/(1+NP.exp(-x))
    def sigmoid_moshtagh(self,output):
        return output*(1-output)
    def feedforward(self,inputs):
        self.inputs=inputs
        self.output=self.sigmoid(NP.dot(inputs,self.weight)+self.bias)
        return self.output
