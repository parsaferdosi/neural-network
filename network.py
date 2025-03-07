import numpy as np
from layers import Layers  
import os 

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.layers = [
            Layers(layer_sizes[i + 1], layer_sizes[i], activation='relu' if i < len(layer_sizes) - 2 else 'sigmoid')
            for i in range(len(layer_sizes) - 1)
        ]
        self.learning_rate = learning_rate 
        


    def feedforward(self, inputs):
        """ اجرای مرحله feedforward در تمام لایه‌ها """
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, train_inputs, train_outputs, epochs):
        """ آموزش شبکه با الگوریتم پس‌انتشار خطا """
        for epoch in range(epochs):
            output = self.feedforward(train_inputs)
            error = train_outputs - output

            for layer in reversed(self.layers):
                error = layer.backward(error, self.learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(error ** 2)  # خطای MSE
                print(f"Epoch {epoch}, Loss: {loss:.5f}")
    def save_weight(self,filename="memmoryCore.npz"):
        weights_data={}
        for i , Layer in enumerate(self.layers):
            weights_data[f"w{i}"]=np.array([neuron.weight for neuron in Layer.neurons])
            weights_data[f"b{i}"]=np.array([neuron.bias for neuron in Layer.neurons])
        np.savez(filename, **weights_data)
        print("train succsesfully saved")
    def load_weight(self,filename="memmoryCore.npz"):
            data=np.load(filename,allow_pickle=True)
            for i,layer in enumerate(self.layers):
                for j, neuron in enumerate(layer.neurons):
                    neuron.weight=data[f"w{i}"][j]
                    neuron.bias=data[f"b{i}"][j]
            print("load complete")
np.random.seed(1)
# **✅ آزمایش
# **✅ داده‌های آموزشی چند نمونه‌ای**
train_inputs = np.array([
    [0.5, -0.6, 0.1, 0.2],
    [0.2,  0.8, -0.5, -0.1],
    [0.9, -0.4,  0.3,  0.7],
    [-0.2, 0.5, 0.9, -0.4]
])
train_outputs = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

# **✅ ایجاد شبکه و آموزش**
layer_sizes = [4, 3, 2]  
network = NeuralNetwork(layer_sizes, learning_rate=0.05)
if os.path.exists("memmoryCore.npz"):
    network.load_weight()
else:
    network.train(train_inputs, train_outputs, epochs=1000)
    network.save_weight()

# **✅ تست شبکه با ورودی‌های جدید**
test_input = np.array([[0.5, -0.6, 0.1, 0.2]])
output = network.feedforward(test_input)
print("output:", output)
