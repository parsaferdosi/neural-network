import numpy as np

class Neuron:
    def __init__(self, input_size, activation="relu"):
        self.weight = np.random.randn(input_size) * np.sqrt(2 / input_size)  # مقداردهی He
        self.bias = np.zeros(1)
        self.activation = activation
        self.raw_input = None  # ذخیره مقدار خام ورودی برای محاسبات گرادیان

    def activate(self, x):
        """ اعمال تابع فعال‌سازی بر روی ورودی """
        self.raw_input = x  # ذخیره مقدار خام برای گرادیان‌گیری
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # جلوگیری از overflow
        elif self.activation == "relu":
            return np.maximum(0, x)
        return x  # خروجی خطی (مناسب برای لایه‌های میانی)

    def activate_derivative(self, x):
        """ محاسبه مشتق تابع فعال‌سازی """
        if self.activation == "sigmoid":
            return x * (1 - x)  # مشتق سیگموید
        elif self.activation == "relu":
            return np.where(self.raw_input > 0, 1, 0)  # مشتق ReLU بر اساس مقدار خام
        return np.ones_like(x)  # مشتق تابع خطی

    def feedforward(self, inputs):
        self.inputs = inputs
        self.output = self.activate(np.dot(inputs, self.weight) + self.bias)[:, np.newaxis]
        return self.output

    def backward(self, error, learning_rate):
        delta = error * self.activate_derivative(self.output)  
        self.weight += learning_rate * np.dot(self.inputs.T, delta)  
        self.bias += learning_rate * np.sum(delta, axis=0)
        return np.dot(delta, self.weight.reshape(1, -1))
