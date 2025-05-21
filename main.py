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
    
    # پیش پردازش عکس سودوکو
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    edges = cv2.Canny(thresh, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    sudoku_contour = None
    img_area = img.shape[0] * img.shape[1]  # مساحت کل تصویر

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        area = cv2.contourArea(contour)

        # فقط کانتورهایی که چهارضلعی هستند و مساحت بزرگی دارند
        if len(approx) == 4 and area > 0.2 * img_area:  # حداقل 20٪ مساحت تصویر
            if area > max_area:
                max_area = area
                sudoku_contour = approx

    if sudoku_contour is None:
        print("خطا: هیچ چهارضلعی‌ای پیدا نشد.")
        return False



print("sudoku solver with opencv :)")
img_path = "images/sudoku.jpg"
main(img_path)