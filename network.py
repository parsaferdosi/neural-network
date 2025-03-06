import numpy as np
from layers import Layer  

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.1):
        self.layers = [Layer(layer_sizes[i + 1], layer_sizes[i]) for i in range(len(layer_sizes) - 1)]
        self.learning_rate = learning_rate  

    def feedforward(self, inputs):
        """ اجرای مرحله feedforward در تمام لایه‌ها """
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs

    def train(self, train_inputs, train_outputs, epochs):
        """ آموزش شبکه با الگوریتم پس‌انتشار خطا """
        for epoch in range(epochs):
            # **فوروارد پس**
            output = self.feedforward(train_inputs)

            # **محاسبه خطای خروجی**
            error = train_outputs - output

            # **پس‌انتشار خطا (Backpropagation)**
            for layer in reversed(self.layers):
                error = layer.backward(error, self.learning_rate)  # انتشار خطا و آپدیت وزن‌ها

            # **چاپ مقدار خطا هر 100 دوره**
            if epoch % 100 == 0:
                loss = np.mean(error ** 2)  # خطای MSE
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

# **🔹 تست شبکه عصبی 🔹**
layer_sizes = [4, 3, 2]  # ورودی: ۴ نورون، لایه پنهان: ۳ نورون، خروجی: ۲ نورون
network = NeuralNetwork(layer_sizes)

# **✅ داده‌های آموزشی**
train_inputs = np.array([[0.5, -0.6, 0.1, 0.2]])  # یک نمونه ورودی
train_outputs = np.array([[1, 0]])  # خروجی هدف

# **✅ آموزش شبکه**
network.train(train_inputs, train_outputs, epochs=1000)

# **✅ تست خروجی پس از آموزش**
output = network.feedforward(train_inputs)
print("خروجی شبکه بعد از آموزش:", output)
