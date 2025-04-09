import cupy as cp  # استفاده از CuPy برای پردازش سریع روی GPU
import numpy as np  # استفاده از NumPy برای برخی پردازش‌های تصویر
import cv2  # OpenCV برای پردازش تصویر
import random  # تولید اعداد تصادفی
import os  # برای مدیریت فایل‌ها و پوشه‌ها
from network import NeuralNetwork  # ایمپورت کلاس شبکه عصبی

# ✅ مسیر فایل داده‌های آموزشی
csv_path = r"E:\My nurual network from scratch\mnist_data\numbers.csv"
image_base_path = r"E:\My nurual network from scratch\mnist_data\numbers"  # مسیر اصلی تصاویر

# ✅ بارگذاری داده‌های آموزشی از CSV
image_paths = []
labels = []

with open(csv_path, "r") as file:
    next(file)  # رد کردن هدر CSV
    for line in file:
        origin, group, label, img_name = line.strip().split(",")
        label = int(label)  # برچسب عددی از ستون label
        image_paths.append(os.path.join(image_base_path, img_name))  # اضافه کردن مسیر کامل تصویر
        labels.append(label)

# ✅ بارگذاری تصویر (این کار باید روی CPU انجام شود چون OpenCV با NumPy سازگار است)
def load_image(image_path, target_size=(28, 28)):
    if not os.path.exists(image_path):  # بررسی وجود تصویر
        print(f"Image not found: {image_path}")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # خواندن تصویر به صورت خاکستری
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None
    img = cv2.resize(img, target_size)  # تغییر اندازه تصویر
    img = img.astype(np.float32) / 255.0  # نرمال‌سازی تصویر
    return cp.array(img)  # بازگشت به صورت آرایه CuPy

# ✅ تبدیل لیبل‌ها به فرمت One-Hot Encoding
def one_hot_encoding(labels):
    labels = cp.array(labels, dtype=cp.int32)  # تبدیل لیبل‌ها به نوع صحیح
    return cp.eye(10, dtype=cp.float32)[labels]

# ✅ تغییر اندازه تصاویر (این کار باید روی CPU انجام شود، چون OpenCV با NumPy سازگار است)
def resize_images(images, new_size=(28, 28)):
    """
    تغییر اندازه تصاویر به ابعاد جدید
    - images: آرایه‌ای شامل تصاویر ورودی
    - new_size: ابعاد جدید تصویر (پیش‌فرض: 28x28)
    خروجی: تصاویر تغییر اندازه داده شده به‌صورت آرایه CuPy
    """
    images_np = cp.asnumpy(images)  # تبدیل CuPy به NumPy برای پردازش با OpenCV
    resized_images = np.zeros((images.shape[0], new_size[0] * new_size[1]), dtype=np.float32)  # ایجاد آرایه خروجی
    
    for i, img in enumerate(images_np):
        img_resized = cv2.resize(img.reshape(28, 28), new_size).flatten()  # تغییر اندازه و تخت کردن تصویر
        resized_images[i] = img_resized  # ذخیره تصویر تغییر یافته
    
    return cp.array(resized_images)  # تبدیل دوباره به CuPy برای پردازش روی GPU

# ✅ بارگذاری و پیش‌پردازش داده‌ها
def load_and_process_data(image_paths, labels):
    images = []
    for img_path in image_paths:
        img = load_image(img_path)  # بارگذاری تصویر
        if img is not None:
            images.append(img)
    
    images = cp.array(images)  # تبدیل تصاویر به CuPy array
    images_resized = resize_images(images)  # تغییر اندازه تصاویر
    labels_onehot = one_hot_encoding(labels)  # تبدیل برچسب‌ها به One-Hot encoding
    return images_resized, labels_onehot

# ✅ پیش‌پردازش داده‌ها
train_images_resized, train_labels_onehot = load_and_process_data(image_paths, labels)

# ✅ افزایش داده‌ها با چرخش و نویز تصادفی
def augment_image(image):
    """
    افزایش داده‌ها با استفاده از چرخش تصادفی و نویز
    - image: تصویر ورودی به‌صورت آرایه 1D (flatten)
    خروجی: تصویر افزایش‌یافته به‌صورت CuPy
    """
    img_np = cp.asnumpy(image.reshape(28, 28))  # تبدیل CuPy به NumPy برای پردازش با OpenCV

    # 🔄 چرخش تصادفی تصویر در محدوده [-10, 10] درجه
    angle = random.uniform(-10, 10)
    matrix = cv2.getRotationMatrix2D((14, 14), angle, 1)  # ایجاد ماتریس چرخش
    img_np = cv2.warpAffine(img_np, matrix, (28, 28), borderMode=cv2.BORDER_REPLICATE)  # اعمال چرخش

    # 🎭 اضافه کردن نویز گوسی با احتمال 30٪
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.1, img_np.shape)  # تولید نویز گوسی
        img_np = np.clip(img_np + noise, 0, 1)  # اعمال نویز و محدود کردن مقدار پیکسل‌ها بین 0 و 1

    return cp.array(img_np.flatten())  # تبدیل دوباره به CuPy و بازگرداندن تصویر افزایش‌یافته

# ✅ ایجاد مجموعه داده افزایش‌یافته
augmented_images = []
augmented_labels = []

for img, label in zip(cp.asnumpy(train_images_resized), cp.asnumpy(train_labels_onehot)):  # تبدیل داده‌ها به NumPy
    augmented_images.append(img)  # ذخیره تصویر اصلی
    augmented_images.append(cp.asnumpy(augment_image(cp.array(img) * 255)) / 255.0)  # ذخیره تصویر افزایش‌یافته
    augmented_labels.append(label)  # ذخیره لیبل اصلی
    augmented_labels.append(label)  # ذخیره لیبل افزایش‌یافته

# ✅ تبدیل لیست‌های افزایش داده‌شده به آرایه‌های NumPy، سپس به CuPy
augmented_images = cp.array(np.array(augmented_images, dtype=np.float32))  # تبدیل به آرایه CuPy
augmented_labels = cp.array(np.array(augmented_labels, dtype=np.int32))  # تبدیل به آرایه CuPy

# ✅ ساخت شبکه عصبی با معماری مشخص‌شده
network = NeuralNetwork(layer_sizes=[784, 576, 128, 64, 10], learning_rate=0.001)

# ✅ تنظیم تایمر برای محاسبه زمان آموزش
start = cp.cuda.Event()  # شروع تایمر
end = cp.cuda.Event()
start.record()

# ✅ شروع آموزش مدل
network.train(augmented_images, augmented_labels, epochs=1000)

# ✅ پایان تایمر و محاسبه مدت زمان آموزش
end.record()
end.synchronize()
training_time = cp.cuda.get_elapsed_time(start, end) / 1000  # تبدیل زمان از میلی‌ثانیه به ثانیه

# ✅ ذخیره وزن‌های آموزش‌یافته مدل
network.save_weights()
print(f"🚀 Training completed in {training_time:.2f} seconds")  # نمایش مدت زمان آموزش
