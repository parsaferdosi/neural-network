import numpy as np
from scipy.ndimage import zoom
from network import NeuralNetwork 
import time
#تابع تغییر اندازه
def resize_images(images, new_size=(12, 12)):
    return np.array([zoom(img.reshape(28, 28), (new_size[0] / 28, new_size[1] / 28)).flatten() for img in images])
start=time.time()
# مدل ذخیره‌شده را بارگذاری کن
model = NeuralNetwork(layer_sizes=[144,36,24, 10], learning_rate=0.001)
model.load_weight("memmoryCore.npz")

# داده‌های تست را بارگذاری کن
test_data = np.loadtxt("mnist_data/mnist_test.csv", delimiter=",")
test_images = test_data[:, 1:] / 255.0  # نرمال‌سازی به بازه [0,1]
test_labels = test_data[:, 0].astype(int)
test_images_resized = resize_images(test_images)  # تبدیل اندازه
test_images = test_images / 255.0 #نرمالسازی داده ها

# اجرای تست
predictions = model.feedforward(test_images_resized)

# یافتن بیشترین مقدار در خروجی شبکه (برچسب پیش‌بینی‌شده)
predicted_labels = np.argmax(predictions, axis=1)

# محاسبه دقت مدل
accuracy = np.mean(predicted_labels == test_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
end=time.time()
print(end-start)