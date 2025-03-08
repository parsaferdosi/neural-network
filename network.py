import numpy as np
from layers import Layers  

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.layers = [
            Layers(layer_sizes[i + 1], layer_sizes[i], activation='relu' if i < len(layer_sizes) - 2 else 'softmax')
            for i in range(len(layer_sizes) - 1)
        ]
        self.learning_rate = learning_rate 
        
    def feedforward(self, inputs):
        """ اجرای مرحله feedforward در تمام لایه‌ها """
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, train_inputs, train_outputs, epochs, batch_size=128):
        num_samples = train_inputs.shape[0]

        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                batch_inputs = train_inputs[i:i+batch_size]
                batch_outputs = train_outputs[i:i+batch_size]

                output = np.squeeze(self.feedforward(batch_inputs))
                error = batch_outputs - output  

                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(train_outputs, self.feedforward(train_inputs))
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def cross_entropy_loss(self, y_true, y_pred):
        """ تابع خطای Cross-Entropy """
        y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)  # جلوگیری از log(0)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    def save_weight(self, filename="memmoryCore.npz"):
        weights_data = {}
        for i, layer in enumerate(self.layers):
            weights_data[f"w{i}"] = np.array([neuron.weight for neuron in layer.neurons])
            weights_data[f"b{i}"] = np.array([neuron.bias[0] for neuron in layer.neurons])  # ذخیره به‌صورت اسکالر
        np.savez(filename, **weights_data)
        print("✅ Weights successfully saved!")

    def load_weight(self, filename="memmoryCore.npz"):
        data = np.load(filename, allow_pickle=True)
        for i, layer in enumerate(self.layers):
            for j, neuron in enumerate(layer.neurons):
                neuron.weight = data[f"w{i}"][j].copy()
                neuron.bias = np.array([data[f"b{i}"][j]])  # تبدیل مقدار اسکالر به آرایه
        print("✅ Weights successfully loaded!")
