import cupy as cp

class Layers:
    def __init__(self, num_neurons, input_size, activation="relu"):
        self.activation = activation
        self.weights = cp.random.randn(input_size, num_neurons) * cp.sqrt(2 / input_size)  # مقداردهی He
        self.biases = cp.zeros((1, num_neurons))

    @staticmethod
    def softmax(x):
        max_x = cp.max(x, axis=1, keepdims=True)
        exp_x = cp.exp(x - max_x)
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)

    def activate(self, x):
        if self.activation == "relu":
            return cp.maximum(0, x)
        elif self.activation == "sigmoid":
            return 1 / (1 + cp.exp(-cp.clip(x, -500, 500)))
        return x  # خروجی خطی

    def activate_derivative(self, x):
        if self.activation == "relu":
            return (x > 0).astype(cp.float32)
        elif self.activation == "sigmoid":
            return x * (1 - x)
        return cp.ones_like(x)

    def feedforward(self, inputs):
        self.inputs = inputs
        self.raw_inputs = cp.dot(inputs, self.weights) + self.biases

        if self.activation == "softmax":
            self.output = self.softmax(self.raw_inputs)
        else:
            self.output = self.activate(self.raw_inputs)

        return self.output

    def backward(self, error, learning_rate):
        if self.activation == "softmax":
            error = self.output * (error - cp.sum(error * self.output, axis=1, keepdims=True))

        activation_derivative = self.activate_derivative(self.raw_inputs)
        delta = error * activation_derivative

        batch_size = delta.shape[0]

        prev_error = cp.dot(delta, self.weights.T)
        self.weights += learning_rate * cp.dot(self.inputs.T, delta) / batch_size
        self.biases += learning_rate * cp.mean(delta, axis=0, keepdims=True)

        cp.get_default_memory_pool().free_all_blocks()

        return prev_error
    def set_weights(self, weights, biases):
        self.weights = weights
        self.biases = biases