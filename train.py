import numpy as np
from scipy.ndimage import zoom
import cv2
import random
from network import NeuralNetwork
import time

start = time.time()

# مسیر فایل داده‌های آموزشی
train_path = r"E:\My nurual network from scratch\mnist_data\mnist_train.csv"

# بارگذاری داده‌ها
train_data = np.loadtxt(train_path, delimiter=',', encoding="latin-1")
train_images = train_data[:, 1:]  # تصاویر
train_labels = train_data[:, 0]   # برچسب‌ها

# **✅ افزایش اندازه تصویر به 24×24**
def resize_images(images, new_size=(24, 24)):
    resized_images = np.array([
        zoom(img.reshape(28, 28), (new_size[0] / 28, new_size[1] / 28), order=1).flatten()
        for img in images
    ])
    return resized_images

# **✅ ایجاد داده‌های چرخش‌یافته و نویزدار**
def augment_image(image):
    """چرخش و تغییر کوچک برای بهبود یادگیری"""
    img = image.reshape(28, 28)

    # چرخش تصادفی بین -10 تا +10 درجه
    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1)
    img = cv2.warpAffine(img, matrix, (28, 28), borderMode=cv2.BORDER_REPLICATE)

    # اضافه کردن نویز در 30٪ مواقع
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.1, img.shape)
        img = np.clip(img + noise, 0, 1)
    
    return img.flatten()

# تغییر اندازه و نرمال‌سازی تصاویر
train_images_resized = resize_images(train_images) / 255.0

# **✅ افزایش داده‌ها: هر تصویر 2 بار تغییر کند (چرخش و نویز)**
augmented_images = []
augmented_labels = []
for img, label in zip(train_images_resized, train_labels):
    augmented_images.append(img)  # تصویر اصلی
    augmented_images.append(augment_image(img * 255) / 255.0)  # تصویر چرخش‌یافته و نویزدار
    augmented_labels.append(label)
    augmented_labels.append(label)

# **تبدیل داده‌ها به آرایه NumPy**
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# **✅ تبدیل لیبل‌ها به One-Hot Encoding**
train_labels_onehot = np.eye(10)[augmented_labels.astype(int)].astype(np.float32)

# بررسی شکل داده‌ها
print(f"Train resized: {augmented_images.shape}")  # انتظار داریم تعداد داده‌ها 2 برابر شده باشد
print(f"Shape of train_labels_onehot: {train_labels_onehot.shape}")

# **🚀 ساخت شبکه عصبی**
network = NeuralNetwork(layer_sizes=[576, 128, 64, 10], learning_rate=0.001)

# **🔄 آموزش مدل**
network.train(augmented_images, train_labels_onehot, epochs=1000)

# **💾 ذخیره وزن‌ها**
network.save_weight()

end = time.time()
print(f"Training time: {end - start:.2f} seconds")
