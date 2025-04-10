from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter,QPen,QMouseEvent , QPixmap,QImage
from PyQt6.QtCore import Qt,QPoint
import cv2
import numpy as np
class Canvas(QWidget):#ایجاد کلاس بوم نقاشی
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setFixedSize(280,280)#اندازه بوم
        self.pixmap=QPixmap(self.size())#تصویر پس زمینه
        self.pixmap.fill(Qt.GlobalColor.white)#پس زمینه سفید
        self.last_point=None#ذخیره نقطه قبلی برای رسم خطوط
        
    def mousePressEvent(self,event:QMouseEvent):
        if event.button()==Qt.MouseButton.LeftButton:
            self.last_point=event.position().toPoint()#ذخیره موقعیت اولیه موس
            
    def mouseMoveEvent(self,event:QMouseEvent):
        if self.last_point is not None:
            painter=QPainter(self.pixmap)
            pen=QPen(Qt.GlobalColor.black,10,Qt.PenStyle.SolidLine,Qt.PenCapStyle.RoundCap,Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point,event.position().toPoint())
            self.last_point=event.position().toPoint()
            self.update()
    def mouseReleaseEvent(self, event:QMouseEvent):
        if event.button()==Qt.MouseButton.LeftButton:
            self.last_point=None
    def paintEvent(self, event):
        #رسم تصویر ذخیره شده روی بوم
        painter=QPainter(self)
        painter.drawPixmap(0,0,self.pixmap)
    def clear(self):
        #پاک کردن بوم نقاشی
        self.pixmap.fill(Qt.GlobalColor.white)
        self.update()
    def get_image_array(self):
        """تبدیل بوم نقاشی به آرایه مناسب برای مدل"""
        image = self.pixmap.toImage()
        buffer = image.bits()
        buffer.setsize(image.width() * image.height() * 4)  
        array = np.frombuffer(buffer, dtype=np.uint8).reshape((image.height(), image.width(), 4))  
        gray = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)  

        inverted = cv2.bitwise_not(gray)  # 🔴 وارونه کردن رنگ تصویر
        resized = cv2.resize(inverted, (24, 24), interpolation=cv2.INTER_AREA)  
        normalized = resized / 255.0  

        print("📌 تصویر پردازش‌شده:", normalized.flatten())  # برای دیباگ
        return normalized.flatten().reshape(1, -1)  # تبدیل به آرایه 1×144
