import numpy as NP

def sigmoid(x):
    return 1/(1+NP.exp(-x))

builder_inputs=NP.array([[0,0,1],
                        [1,1,1],
                        [1,0,1],
                        [0,1,1]])
builder_outputs=NP.array([0,1,1,0]).T

NP.random.seed(1)
synaps_weight = 2 * NP.random.random((3, 1)) - 1
print(synaps_weight)

for i in range(1):
    input_layers=builder_inputs
    output=sigmoid(NP.dot(input_layers,synaps_weight))

print("output is:\n",output)