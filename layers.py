import cupy as cp  # استفاده از کتابخانه CuPy برای پردازش سریع‌تر روی GPU

class Layers:
    def __init__(self, num_neurons, input_size, activation="relu"):
        """
        مقداردهی اولیه لایه:
        - num_neurons: تعداد نورون‌های لایه
        - input_size: تعداد ورودی‌های هر نورون
        - activation: تابع فعال‌سازی (پیش‌فرض: ReLU)
        """
        self.activation = activation
        # مقداردهی اولیه وزن‌ها با روش He برای بهبود همگرایی مدل
        self.weights = cp.random.randn(input_size, num_neurons) * cp.sqrt(2 / input_size)
        self.biases = cp.zeros((1, num_neurons))  # مقداردهی اولیه بایاس‌ها به صفر

    @staticmethod
    def softmax(x):
        """
        محاسبه تابع Softmax برای تبدیل مقادیر ورودی به احتمال دسته‌ها
        """
        max_x = cp.max(x, axis=1, keepdims=True)  # جلوگیری از overflow با کاهش بیشترین مقدار
        exp_x = cp.exp(x - max_x)  
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)  # نرمال‌سازی خروجی‌ها

    def activate(self, x):
        """
        اعمال تابع فعال‌سازی بر روی ورودی
        """
        if self.activation == "relu":
            return cp.maximum(0, x)  # ReLU
        elif self.activation == "sigmoid":
            return 1 / (1 + cp.exp(-cp.clip(x, -500, 500)))  # سیگموید با محدودسازی مقدار برای جلوگیری از overflow
        return x  # خروجی خطی (برای لایه‌های بدون تابع فعال‌سازی)

    def activate_derivative(self, x):
        """
        محاسبه مشتق تابع فعال‌سازی برای استفاده در پس‌انتشار خطا
        """
        if self.activation == "relu":
            return (x > 0).astype(cp.float32)  # مشتق ReLU: 1 برای مقادیر مثبت و 0 برای مقادیر منفی
        elif self.activation == "sigmoid":
            return x * (1 - x)  # مشتق سیگموید
        return cp.ones_like(x)  # برای توابع خطی مقدار مشتق برابر 1 است

    def feedforward(self, inputs):
        """
        اجرای محاسبات رو به جلو (Feedforward)
        - ضرب ورودی در وزن‌ها و افزودن بایاس
        - اعمال تابع فعال‌سازی
        """
        self.inputs = inputs
        self.raw_inputs = cp.dot(inputs, self.weights) + self.biases  # محاسبه مقدار ورودی خام نورون‌ها

        if self.activation == "softmax":
            self.output = self.softmax(self.raw_inputs)  # اعمال Softmax برای لایه خروجی
        else:
            self.output = self.activate(self.raw_inputs)  # اعمال تابع فعال‌سازی معمولی

        return self.output  # بازگرداندن خروجی لایه

    def backward(self, error, learning_rate):
        """
        اجرای پس‌انتشار خطا (Backpropagation)
        - محاسبه گرادیان تابع هزینه نسبت به وزن‌ها و بایاس‌ها
        - به‌روزرسانی وزن‌ها و بایاس‌ها با استفاده از نرخ یادگیری
        """
        if self.activation == "softmax":
            # محاسبه گرادیان تابع هزینه برای softmax
            error = self.output * (error - cp.sum(error * self.output, axis=1, keepdims=True))

        activation_derivative = self.activate_derivative(self.raw_inputs)  # مشتق تابع فعال‌سازی
        delta = error * activation_derivative  # محاسبه گرادیان خروجی نورون‌ها

        batch_size = delta.shape[0]  # اندازه بچ داده

        prev_error = cp.dot(delta, self.weights.T)  # خطای لایه قبلی برای پس‌انتشار بیشتر
        self.weights += learning_rate * cp.dot(self.inputs.T, delta) / batch_size  # به‌روزرسانی وزن‌ها
        self.biases += learning_rate * cp.mean(delta, axis=0, keepdims=True)  # به‌روزرسانی بایاس‌ها

        cp.get_default_memory_pool().free_all_blocks()  # آزادسازی حافظه GPU برای جلوگیری از اشغال شدن غیرضروری

        return prev_error  # بازگرداندن خطای لایه برای پس‌انتشار به لایه قبلی

    def set_weights(self, weights, biases):
        """
        تنظیم وزن‌ها و بایاس‌های لایه از مقدار مشخص‌شده
        """
        self.weights = weights
        self.biases = biases
