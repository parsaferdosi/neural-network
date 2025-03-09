import cupy as cp
import numpy as np
from layers import Layers  

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.layers = [Layers(layer_sizes[i+1], layer_sizes[i], 'relu' if i < len(layer_sizes)-2 else 'softmax')
                       for i in range(len(layer_sizes) - 1)]
        self.learning_rate = learning_rate 

    def feedforward(self, inputs):
        inputs = cp.asarray(inputs)
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, train_inputs, train_outputs, epochs, batch_size=128):
        train_inputs = cp.asarray(train_inputs)
        train_outputs = cp.asarray(train_outputs)

        num_samples = train_inputs.shape[0]

        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                batch_inputs = train_inputs[i:i+batch_size]
                batch_outputs = train_outputs[i:i+batch_size]

                output = self.feedforward(batch_inputs)
                error = batch_outputs - output  

                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            cp.get_default_memory_pool().free_all_blocks()

            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(train_outputs, self.feedforward(train_inputs))
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def cross_entropy_loss(self, y_true, y_pred):
            y_pred = cp.clip(y_pred, 1e-9, 1 - 1e-9)
            return -cp.sum(y_true * cp.log(y_pred)) / y_true.shape[0]
    def save_weights(self, filename="memoryCore.npz"):
        weights_dict = {}

        for i, layer in enumerate(self.layers):
            weights_dict[f"weights_{i}"] = layer.weights.get()  # cupy → numpy
            weights_dict[f"biases_{i}"] = layer.biases.get()  # cupy → numpy

        np.savez_compressed(filename, **weights_dict)
        
        print(f"✅ Weights saved successfully to {filename}")

    def load_weights(self, filename="memmoryCore.npz"):
        try:
            data = cp.load(filename, allow_pickle=True)
            for i, layer in enumerate(self.layers):
                layer.set_weights((cp.asarray(data[f"arr_{i}"][0]), cp.asarray(data[f"arr_{i}"][1])))
            print(f"✅ Weights loaded from {filename}")
        except FileNotFoundError:
            print("❌ No saved weights found!")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")

