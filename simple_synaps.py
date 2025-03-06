import numpy as NP

def sigmoid(x):
    return 1 / (1 + NP.exp(-x))

def sigmoid_moshtagh(x):
    return x * (1 - x)

builder_inputs = NP.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]]) # این داده‌های آموزشی است
builder_outputs = NP.array([[0, 1, 1, 0]]).T # خروجی داده‌های آموزشی

NP.random.seed(1)
synaps_weight = 2 * NP.random.random((3, 1)) - 1
print(synaps_weight)

for i in range(20000): # بخش آموزش
    input_layers = builder_inputs
    output = sigmoid(NP.dot(input_layers, synaps_weight))
    error = builder_outputs - output
    adjustment = error * sigmoid_moshtagh(output)
    synaps_weight += NP.dot(input_layers.T, adjustment)

print("weight of synaps after train:\n", synaps_weight)
print("output:\n", output)
