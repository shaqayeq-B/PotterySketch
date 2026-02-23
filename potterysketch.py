import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def generate_bold_ancient_sketch(image_path):
    # ۱. خواندن تصویر
    img = cv2.imread(image_path)
    if img is None:
        print(f"خطا: تصویری در مسیر {image_path} پیدا نشد.")
        return None
    
    # ۲. تبدیل به خاکستری و تقویت کنتراست
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # افزایش تضاد نوری
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # ۳. حذف نویز بدون محو کردن لبه‌ها
    smooth = cv2.medianBlur(contrast, 3)
    
    # ۴. استخراج لبه‌های پررنگ (Adaptive Thresholding)
    bold_edges = cv2.adaptiveThreshold(smooth, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    
    # ۵. ضخیم کردن خطوط (Erosion برای خطوط سیاه در پس‌زمینه سفید)
    kernel = np.ones((2,2), np.uint8)
    bold_edges = cv2.erode(bold_edges, kernel, iterations=1)
    
    # ۶. تمیزکاری نهایی
    final_sketch = cv2.medianBlur(bold_edges, 3)
    
    return final_sketch

# --- اجرا در VS Code ---
# نام فایل عکس خود را که در کنار این اسکریپت قرار دارد، اینجا بنویسید
input_filename = "your_image.jpg" 
output_filename = "bold_ancient_sketch.jpg"

if os.path.exists(input_filename):
    print("در حال تولید اسکچ پررنگ و واضح...")
    result = generate_bold_ancient_sketch(input_filename)
    
    if result is not None:
        # نمایش خروجی در پنجره
        plt.figure(figsize=(10, 10))
        plt.imshow(result, cmap='gray')
        plt.axis("off")
        plt.title("Final Sketch")
        plt.show()
        
        # ذخیره روی هارد
        cv2.imwrite(output_filename, result)
        print(f"فایل با موفقیت ذخیره شد: {output_filename}")
else:
    print(f"لطفاً مطمئن شوید فایل '{input_filename}' در پوشه پروژه قرار دارد.")
