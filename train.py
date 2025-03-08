import numpy as np
from scipy.ndimage import zoom
from network import NeuralNetwork
import time

start = time.time()

# مسیر فایل داده‌های آموزشی
train_path = r"E:\My nurual network from scratch\mnist_data\mnist_train.csv"

# بارگذاری داده‌ها
train_data = np.loadtxt(train_path, delimiter=',', encoding="latin-1")
train_images = train_data[:, 1:]  # تصاویر
train_labels = train_data[:, 0]   # برچسب‌ها

# تابع تغییر اندازه تصاویر
def resize_images(images, new_size=(12, 12)):
    resized_images = np.array([
        zoom(img.reshape(28, 28), (new_size[0] / 28, new_size[1] / 28), order=1).flatten()
        for img in images
    ])
    return resized_images

# تغییر اندازه و نرمال‌سازی
train_images_resized = resize_images(train_images) / 255.0

# تبدیل لیبل‌ها به one-hot encoding
train_labels_onehot = np.eye(10)[train_labels.astype(int)].astype(np.float32)

# بررسی شکل داده‌ها
print(f"Train resized: {train_images_resized.shape}")
print(f"Shape of train_labels_onehot: {train_labels_onehot.shape}")

# ساخت شبکه عصبی
network = NeuralNetwork(layer_sizes=[144,36,24, 10], learning_rate=0.001)

# آموزش مدل
network.train(train_images_resized, train_labels_onehot, epochs=1000)

# ذخیره وزن‌ها
network.save_weight()

end = time.time()
print(f"Training time: {end - start:.2f} seconds")
