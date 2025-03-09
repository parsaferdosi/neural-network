import cupy as cp
import cv2
import random
from network import NeuralNetwork

# ✅ مسیر فایل داده‌های آموزشی
train_path = r"E:\My nurual network from scratch\mnist_data\mnist_train.csv"

# ✅ بارگذاری داده‌ها مستقیماً روی GPU
train_data = cp.loadtxt(train_path, delimiter=',', dtype=cp.float32)
train_images = train_data[:, 1:] / 255.0  # نرمال‌سازی روی GPU
train_labels = train_data[:, 0].astype(cp.int32)

# ✅ تغییر اندازه تصاویر
def resize_images(images, new_size=(28, 28)):
    resized_images = cp.zeros((images.shape[0], new_size[0] * new_size[1]), dtype=cp.float32)
    for i, img in enumerate(images):
        img_cpu = cp.asnumpy(img.reshape(28, 28))  # تبدیل به NumPy برای OpenCV
        img_resized = cv2.resize(img_cpu, new_size).flatten()
        resized_images[i] = cp.array(img_resized)  # تبدیل به CuPy
    return resized_images

train_images_resized = resize_images(train_images)

# ✅ افزایش داده‌ها (چرخش و نویز)
def augment_image(image):
    """چرخش تصادفی و اضافه کردن نویز"""
    img = image.reshape(28, 28)

    # چرخش تصادفی
    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1)
    img = cv2.warpAffine(cp.asnumpy(img), matrix, (28, 28), borderMode=cv2.BORDER_REPLICATE)

    # نویز تصادفی
    if random.random() < 0.3:
        noise = cp.random.normal(0, 0.1, img.shape)
        img = cp.clip(cp.array(img) + noise, 0, 1)

    return img.flatten()

augmented_images = []
augmented_labels = []
for img, label in zip(train_images_resized, train_labels):
    augmented_images.append(img)
    augmented_images.append(augment_image(img * 255) / 255.0)
    augmented_labels.append(label)
    augmented_labels.append(label)

augmented_images = cp.array(augmented_images)
augmented_labels = cp.array(augmented_labels)

# ✅ تبدیل لیبل‌ها به One-Hot Encoding روی GPU
train_labels_onehot = cp.eye(10, dtype=cp.float32)[augmented_labels]

# ✅ بررسی شکل داده‌ها
print(f"Train data shape: {augmented_images.shape}")
print(f"Labels shape: {train_labels_onehot.shape}")

# ✅ ساخت شبکه عصبی
network = NeuralNetwork(layer_sizes=[784, 576, 128, 64, 10], learning_rate=0.001)

# ✅ تنظیم تایمر CuPy
start = cp.cuda.Event()
end = cp.cuda.Event()
start.record()

# ✅ آموزش مدل
network.train(augmented_images, train_labels_onehot, epochs=1)

end.record()
end.synchronize()
training_time = cp.cuda.get_elapsed_time(start, end) / 1000  # تبدیل میلی‌ثانیه به ثانیه

# ✅ ذخیره وزن‌ها
network.save_weights()
print(f"🚀 Training completed in {training_time:.2f} seconds")
