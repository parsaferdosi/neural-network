import cupy as cp  # استفاده از CuPy برای پردازش سریع روی GPU
import numpy as np  # استفاده از NumPy برای برخی پردازش‌های تصویر
import cv2  # OpenCV برای پردازش تصویر
import random  # تولید اعداد تصادفی
from network import NeuralNetwork  # ایمپورت کلاس شبکه عصبی

# ✅ مسیر فایل داده‌های آموزشی
train_path = r"E:\My nurual network from scratch\mnist_data\mnist_train.csv"

# ✅ بارگذاری داده‌های آموزشی روی GPU
train_data = cp.loadtxt(train_path, delimiter=',', dtype=cp.float32)  # خواندن داده‌ها از فایل CSV
train_images = train_data[:, 1:] / 255.0  # استخراج تصاویر و نرمال‌سازی پیکسل‌ها به محدوده [0,1]
train_labels = train_data[:, 0].astype(cp.int32)  # استخراج لیبل‌ها و تبدیل آن‌ها به عدد صحیح

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

# ✅ اعمال تغییر اندازه روی تصاویر آموزشی
train_images_resized = resize_images(train_images)

# ✅ افزایش داده‌ها با چرخش و نویز تصادفی (این کار روی CPU انجام می‌شود)
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

# ✅ ایجاد مجموعه داده افزایش‌یافته (افزایش داده‌ها با دو برابر شدن مجموعه)
augmented_images = []
augmented_labels = []

for img, label in zip(cp.asnumpy(train_images_resized), cp.asnumpy(train_labels)):  # تبدیل داده‌ها به NumPy
    augmented_images.append(img)  # ذخیره تصویر اصلی
    augmented_images.append(cp.asnumpy(augment_image(cp.array(img) * 255)) / 255.0)  # ذخیره تصویر افزایش‌یافته
    augmented_labels.append(label)  # ذخیره لیبل اصلی
    augmented_labels.append(label)  # ذخیره لیبل افزایش‌یافته

# ✅ تبدیل لیست‌های افزایش داده‌شده به آرایه‌های NumPy، سپس به CuPy
augmented_images = cp.array(np.array(augmented_images, dtype=np.float32))  # تبدیل به آرایه CuPy
augmented_labels = cp.array(np.array(augmented_labels, dtype=np.int32))  # تبدیل به آرایه CuPy

# ✅ تبدیل لیبل‌ها به فرمت One-Hot Encoding برای یادگیری بهتر
train_labels_onehot = cp.eye(10, dtype=cp.float32)[augmented_labels]

# ✅ بررسی شکل داده‌های پردازش‌شده
print(f"Train data shape: {augmented_images.shape}")  # نمایش تعداد نمونه‌های نهایی
print(f"Labels shape: {train_labels_onehot.shape}")  # نمایش شکل برچسب‌های One-Hot

# ✅ ساخت شبکه عصبی با معماری مشخص‌شده
network = NeuralNetwork(layer_sizes=[784, 576, 128, 64, 10], learning_rate=0.001)

# ✅ تنظیم تایمر برای محاسبه زمان آموزش
start = cp.cuda.Event()  # شروع تایمر
end = cp.cuda.Event()
start.record()

# ✅ شروع آموزش مدل
network.train(augmented_images, train_labels_onehot, epochs=1000)

# ✅ پایان تایمر و محاسبه مدت زمان آموزش
end.record()
end.synchronize()
training_time = cp.cuda.get_elapsed_time(start, end) / 1000  # تبدیل زمان از میلی‌ثانیه به ثانیه

# ✅ ذخیره وزن‌های آموزش‌یافته مدل
network.save_weights()
print(f"🚀 Training completed in {training_time:.2f} seconds")  # نمایش مدت زمان آموزش
