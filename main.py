from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import tensorflow as tf
import argparse



@dataclass
class ImageTransform:
    """Stores image transformation data."""
    warped: np.ndarray
    side: int
    contour: np.ndarray
    resized_img: np.ndarray


class SudokuExtractor:
    """Extracts digits from a Sudoku image, solves it, and generates a visual output."""

    def __init__(self, model_path: str, output_dir: str = "cells"):
        """Initialize with a trained model and output directory."""
        self.model = tf.keras.models.load_model(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.original_grid = None

    def _preprocess_cell(self, cell: np.ndarray, row: int, col: int) -> np.ndarray:
        """Preprocess a single cell image for digit recognition."""
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        height, width = thresh.shape
        border = int(min(height, width) * 0.1)
        thresh = thresh[border:height - border, border:width - border]
        thresh = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
        thresh = thresh.astype('float32') / 255.0

        cv2.imwrite(str(self.output_dir / f"processed_cell_{row}_{col}.jpg"), thresh * 255)
        return np.expand_dims(thresh, axis=-1)

    def _extract_number(self, cell: np.ndarray, row: int, col: int) -> int:
        """Extract a digit from a cell image."""
        processed_cell = self._preprocess_cell(cell, row, col)
        if np.mean(processed_cell) < 0.05:
            return 0

        input_image = np.expand_dims(processed_cell, axis=0)
        prediction = self.model.predict(input_image, verbose=0)
        confidence = np.max(prediction)
        number = np.argmax(prediction, axis=1)[0]

        return 0 if confidence < 0.7 and np.mean(processed_cell) < 0.1 else int(number)

    def _remove_grid_lines(self, warped: np.ndarray) -> np.ndarray:
        """Remove grid lines from the Sudoku image."""
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3
        )
        cv2.imwrite(str(self.output_dir / "thresh.jpg"), thresh)

        kernel_h = np.ones((1, 20), np.uint8)
        kernel_v = np.ones((20, 1), np.uint8)
        horizontal = cv2.dilate(thresh, kernel_h, iterations=2)
        vertical = cv2.dilate(thresh, kernel_v, iterations=2)
        grid = cv2.add(horizontal, vertical)
        cleaned = cv2.bitwise_and(thresh, cv2.bitwise_not(grid))
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imwrite(str(self.output_dir / "cleaned_sudoku.jpg"), cleaned)
        return cleaned

    def _find_sudoku_contour(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find the largest quadrilateral contour in the image."""
        height, width = image.shape[:2]
        scale = 800 / max(height, width)
        resized = cv2.resize(image, (int(width * scale), int(height * scale)))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 3
        )
        edges = cv2.Canny(thresh, 30, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        sudoku_contour = None
        img_area = resized.shape[0] * resized.shape[1]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)
            if len(approx) == 4 and area > 0.1 * img_area and area > max_area:
                max_area = area
                sudoku_contour = approx

        if sudoku_contour is None:
            cv2.imwrite(str(self.output_dir / "edges.jpg"), edges)
            raise ValueError("Sudoku grid not detected. Please check the image.")

        return sudoku_contour, resized

    def _warp_image(self, image: np.ndarray, contour: np.ndarray) -> Tuple[np.ndarray, int]:
        """Warp the image to a square perspective based on the contour."""
        points = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0], rect[2] = points[np.argmin(s)], points[np.argmax(s)]
        diff = np.diff(points, axis=1)
        rect[1], rect[3] = points[np.argmin(diff)], points[np.argmax(diff)]

        width_a = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
        width_b = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
        height_a = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
        height_b = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))

        side = max(int(max(width_a, width_b)), int(max(height_a, height_b)))
        dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (side, side))
        cv2.imwrite(str(self.output_dir / "warped_sudoku.jpg"), warped)
        return warped, side

    def _is_valid(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing a number in the given position is valid."""
        if num in grid[row, :]:
            return False
        if num in grid[:, col]:
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in grid[start_row:start_row + 3, start_col:start_col + 3]:
            return False
        return True
    
    def _solve_sudoku(self, grid: np.ndarray) -> bool:
        """Solve the Sudoku puzzle using backtracking."""
        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:
                    for num in range(1, 10):
                        if self._is_valid(grid, row, col, num):
                            grid[row, col] = num
                            if self._solve_sudoku(grid):
                                return True
                            grid[row, col] = 0
                    return False
        return True
    
    def _create_solved_image(self, transform: ImageTransform, grid: np.ndarray) -> None:
        """Create an image with the solved Sudoku grid overlaid."""
        output_image = transform.warped.copy()
        cell_size = transform.side // 9
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = cell_size / 40
        thickness = max(1, int(cell_size / 30))
        original_color = (0, 0, 255)  # Red for original digits
        solved_color = (0, 255, 0)  # Green for solved digits

        for i in range(9):
            for j in range(9):
                if grid[i, j] != 0:
                    x = j * cell_size + int(cell_size * 0.35)
                    y = i * cell_size + int(cell_size * 0.65)
                    color = solved_color if grid[i, j] != self.original_grid[i, j] else original_color
                    cv2.putText(
                        output_image,
                        str(grid[i, j]),
                        (x, y),
                        font,
                        font_scale,
                        color,
                        thickness,
                        cv2.LINE_AA
                    )

        cv2.imwrite(str(self.output_dir / "solved_sudoku.jpg"), output_image)

    def extract_and_solve(self, image_path: str) -> Optional[np.ndarray]:
        """Extract and solve the Sudoku grid from an image, generating a solved image."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found at: {image_path.absolute()}")

            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Failed to load image. Check file format or path.")

            contour, resized_img = self._find_sudoku_contour(img)
            cv2.imwrite(
                str(self.output_dir / "detected_contour.jpg"),
                cv2.drawContours(resized_img.copy(), [contour], 0, (0, 0, 255), 3)
            )

            warped, side = self._warp_image(resized_img, contour)
            cleaned = self._remove_grid_lines(warped)
            cell_size = side // 9
            cleaned = cv2.resize(cleaned, (cell_size * 9, cell_size * 9))
            warped = cv2.resize(warped, (cell_size * 9, cell_size * 9))

            sudoku_grid = np.zeros((9, 9), dtype=int)
            for i in range(9):
                for j in range(9):
                    x_start, y_start = j * cell_size, i * cell_size
                    cell = cleaned[y_start:y_start + cell_size, x_start:x_start + cell_size]
                    cv2.imwrite(str(self.output_dir / f"cell_{i}_{j}.jpg"), cell)
                    sudoku_grid[i, j] = self._extract_number(
                        warped[y_start:y_start + cell_size, x_start:x_start + cell_size], i, j
                    )

            self.original_grid = sudoku_grid.copy()
            print("Original Sudoku grid:")
            print(sudoku_grid)

            if not self._solve_sudoku(sudoku_grid):
                print("Warning: Could not solve the Sudoku puzzle.")
            else:
                print("Sudoku solved successfully!")
                print("Solved Sudoku grid:")
                print(sudoku_grid)

            self._create_solved_image(ImageTransform(warped, side, contour, resized_img), sudoku_grid)
            print(f"Cells and solved image saved to '{self.output_dir}'.")
            return sudoku_grid

        except Exception as e:
            print(f"Error: {e}")
            return None
    

def main():
    """Run the Sudoku extractor and solver."""
    print("Sudoku solver:)")
    extractor = SudokuExtractor("models/digit_models.h5")
        
    parser = argparse.ArgumentParser(
                description="This script help you solve sudoku with computer vision."
            )
    parser.add_argument("image_path", type=str, help="your image location.")
    arg = parser.parse_args()
    
    return extractor.extract_and_solve(arg.image_path)



if __name__ == "__main__":
    main()
