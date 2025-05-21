import cv2
import numpy as np
import os

def main(image_path, output_dir="cells"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = cv2.imread(image_path)
    if img is None:
        print("خطا: تصویر بارگذاری نشد.")
        return False
    
     # تغییر اندازه تصویر برای پردازش بهتر
    height, width = img.shape[:2]
    scale = 800 / max(height, width)
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


print("sudoku solver with opencv :)")
img_path = "images/sudoku.jpg"
main(img_path)