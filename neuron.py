#کلاس نورون که پایه کاره
import numpy as NP
class Neuron:
    def __init__(self,weight,bias):
        self.weight=weight
        self.bias=bias
    def sigmoid(self,x):
        return 1/(1+NP.exp(-x))
    def sigmoid_moshtagh(self,x):
        return x*(1-x)
    def feedforward(self,inputs):
        return self.sigmoid(NP.dot(inputs,self.weight)+self.bias)
    def train(self,train_inputs,train_outputs,epochs):
        for epoch in range(epochs):
            inputs=train_inputs
            output=self.feedforward(inputs)
            error=train_outputs-output
            adjustment=error*self.sigmoid_moshtagh(output)
            self.weight += NP.dot(inputs.T,adjustment)
            self.bias+=NP.sum(error*self.sigmoid_moshtagh(output),axis=0)

