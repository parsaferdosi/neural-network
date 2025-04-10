import sys
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt
from canvans import Canvas  # فایل بوم نقاشی
from network import NeuralNetwork
class MainWindow(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("تشخیص عدد")
        self.setFixedSize(350, 400)

        self.model = model  # مدل شبکه عصبی

        # ایجاد ویجت‌های رابط کاربری
        self.canvas = Canvas(self)
        self.result_label = QLabel("نتیجه: ?", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.recognize_button = QPushButton("تشخیص عدد", self)
        self.clear_button = QPushButton("پاک کردن", self)

        # تنظیمات لایه‌بندی
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.result_label)
        layout.addWidget(self.recognize_button)
        layout.addWidget(self.clear_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # متصل کردن دکمه‌ها به توابع
        self.recognize_button.clicked.connect(self.recognize_digit)
        self.clear_button.clicked.connect(self.canvas.clear)

    def recognize_digit(self):
        image_array = self.canvas.get_image_array()
        
        print("proccessed image:")
        print(image_array)  # چاپ ورودی برای دیباگ
        
        prediction = self.model.feedforward(image_array)
        predicted_digit = np.argmax(prediction)
        self.result_label.setText(f"نتیجه: {predicted_digit}")
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # بارگذاری مدل
    model = NeuralNetwork(layer_sizes=[576, 128, 64, 10], learning_rate=0.001)
    model.load_weight("memmoryCore.npz")  # بارگذاری وزن‌های ذخیره‌شده
    
    window = MainWindow(model)
    window.show()

    sys.exit(app.exec())

