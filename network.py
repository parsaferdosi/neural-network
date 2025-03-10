import cupy as cp  # استفاده از CuPy برای پردازش روی GPU
import numpy as np  # استفاده از NumPy برای ذخیره و بازیابی وزن‌ها
from layers import Layers  # ایمپورت کلاس لایه‌ها

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        """
        مقداردهی اولیه شبکه عصبی:
        - layer_sizes: لیستی شامل تعداد نورون‌های هر لایه
        - learning_rate: نرخ یادگیری برای به‌روزرسانی وزن‌ها
        """
        self.layers = [Layers(layer_sizes[i+1], layer_sizes[i], 'relu' if i < len(layer_sizes)-2 else 'softmax')
                       for i in range(len(layer_sizes) - 1)]  # ایجاد لایه‌ها با ReLU در لایه‌های مخفی و Softmax در خروجی
        self.learning_rate = learning_rate  # ذخیره نرخ یادگیری

    def feedforward(self, inputs):
        """
        اجرای فرآیند پیش‌روی (Feedforward) برای دریافت خروجی مدل
        - inputs: ورودی‌های شبکه به‌صورت یک آرایه
        """
        inputs = cp.asarray(inputs)  # تبدیل ورودی‌ها به آرایه CuPy برای اجرا روی GPU
        for layer in self.layers:
            inputs = layer.feedforward(inputs)  # عبور داده از هر لایه
        return inputs  # بازگرداندن خروجی نهایی شبکه

    def train(self, train_inputs, train_outputs, epochs, batch_size=128):
        """
        آموزش مدل با استفاده از پس‌انتشار خطا (Backpropagation)
        - train_inputs: داده‌های ورودی آموزش
        - train_outputs: برچسب‌های صحیح داده‌ها
        - epochs: تعداد تکرارها برای آموزش
        - batch_size: اندازه‌ی هر دسته در یادگیری دسته‌ای (Mini-Batch)
        """
        train_inputs = cp.asarray(train_inputs)  # تبدیل ورودی‌ها به CuPy
        train_outputs = cp.asarray(train_outputs)  # تبدیل خروجی‌ها به CuPy

        num_samples = train_inputs.shape[0]  # تعداد کل نمونه‌ها

        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                batch_inputs = train_inputs[i:i+batch_size]  # انتخاب دسته‌ای از داده‌ها
                batch_outputs = train_outputs[i:i+batch_size]

                output = self.feedforward(batch_inputs)  # انجام Feedforward
                error = batch_outputs - output  # محاسبه خطای مدل

                # پس‌انتشار خطا از لایه‌های خروجی به ورودی
                for layer in reversed(self.layers):
                    error = layer.backward(error, self.learning_rate)

            # آزادسازی حافظه برای بهبود عملکرد
            cp.get_default_memory_pool().free_all_blocks()

            # هر 100 دوره یک‌بار، میزان خطا محاسبه و نمایش داده شود
            if epoch % 100 == 0:
                loss = self.cross_entropy_loss(train_outputs, self.feedforward(train_inputs))
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def cross_entropy_loss(self, y_true, y_pred):
        """
        محاسبه مقدار خطای Cross-Entropy برای ارزیابی مدل
        - y_true: خروجی‌های واقعی (One-Hot Encoding)
        - y_pred: خروجی‌های پیش‌بینی‌شده مدل
        """
        y_pred = cp.clip(y_pred, 1e-9, 1 - 1e-9)  # جلوگیری از مقدار صفر در لگاریتم
        return -cp.sum(y_true * cp.log(y_pred)) / y_true.shape[0]  # میانگین خطای دسته‌ای

    def save_weights(self, filename="memoryCore.npz"):
        """
        ذخیره وزن‌ها و بایاس‌های مدل در یک فایل
        - filename: نام فایل ذخیره‌سازی (پیش‌فرض: memoryCore.npz)
        """
        weights_dict = {}

        for i, layer in enumerate(self.layers):
            weights_dict[f"weights_{i}"] = layer.weights.get()  # تبدیل CuPy به NumPy برای ذخیره
            weights_dict[f"biases_{i}"] = layer.biases.get()  # تبدیل CuPy به NumPy

        np.savez_compressed(filename, **weights_dict)  # ذخیره وزن‌ها در قالب فشرده‌شده

        print(f"✅ Weights saved successfully to {filename}")

    def load_weights(self, filename="memoryCore.npz"):
        """
        بارگذاری وزن‌ها و بایاس‌های مدل از فایل ذخیره‌شده
        - filename: نام فایلی که شامل وزن‌های ذخیره‌شده است
        """
        try:
            data = np.load(filename, allow_pickle=True)  # بارگذاری اطلاعات از فایل
            for i, layer in enumerate(self.layers):
                layer.set_weights(cp.asarray(data[f"weights_{i}"]), cp.asarray(data[f"biases_{i}"]))  # تبدیل NumPy به CuPy
            print(f"✅ Weights loaded from {filename}")
        except FileNotFoundError:
            print("❌ No saved weights found!")  # اگر فایل وجود نداشته باشد
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")  # نمایش خطا در صورت بروز مشکل
