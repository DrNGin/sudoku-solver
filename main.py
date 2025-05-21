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
    
    # img_contour = img.copy()
    # cv2.drawContours(img_contour, [sudoku_contour], 0, (0, 0, 255), 3)
    # cv2.imshow("Detected Sudoku Contour", img_contour)
    # cv2.waitKey(0)

    points = sudoku_contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # بالا-چپ
    rect[2] = points[np.argmax(s)]  # پایین-راست
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # بالا-راست
    rect[3] = points[np.argmax(diff)]  # پایین-چپ
    
    # محاسبه ابعاد جدول
    (tl, tr, br, bl) = rect
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # تنظیم ابعاد به مربع
    side = max(max_width, max_height)
    dst = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (side, side))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Warped Sudoku", warped_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

print("sudoku solver with opencv :)")
img_path = "images/sudoku.jpg"
main(img_path)