# Sudoku Solver with Computer Vision

This project uses computer vision and machine learning to extract a Sudoku grid from an image, recognize the digits, solve the puzzle, and generate a visual output with the solved grid.

## Features
- **Image Processing**: Detects and warps the Sudoku grid from an input image using OpenCV.
- **Digit Recognition**: Utilizes a pre-trained TensorFlow model to recognize digits in the grid cells.
- **Sudoku Solving**: Implements a backtracking algorithm to solve the extracted Sudoku puzzle.
- **Output Generation**: Saves intermediate images (e.g., detected contours, processed cells) and a final solved Sudoku image with original digits in red and filled digits in green.

## Requirements
To run this project, you need the following dependencies:
- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy
- TensorFlow
- Argparse (included in Python standard library)

Install the dependencies using:
```bash
pip install opencv-python numpy tensorflow
```

## Usage
1. **Prepare the Model**: Ensure you have a pre-trained TensorFlow model (`digit_models.h5`) for digit recognition. Place it in the `models/` directory.
2. **Run the Script**: Use the command line to provide the path to your Sudoku image:
   ```bash
   python sudoku_extractor.py path/to/your/sudoku_image.jpg
   ```
3. **Output**: The script will:
   - Create a `cells/` directory to store intermediate images (e.g., processed cells, warped grid, etc.).
   - Print the original and solved Sudoku grids to the console.
   - Save the solved Sudoku image as `solved_sudoku.jpg` in the `cells/` directory, with original digits in red and solved digits in green.

## Example
```bash
python sudoku_extractor.py images/sudoku1.jpg
```

**Output**:
- Console output showing the original and solved Sudoku grids.
- A `cells/` directory containing:
  - `detected_contour.jpg`: Image with the detected Sudoku grid contour.
  - `warped_sudoku.jpg`: Warped Sudoku grid.
  - `cleaned_sudoku.jpg`: Grid with lines removed.
  - `processed_cell_{row}_{col}.jpg`: Processed cell images for digit recognition.
  - `solved_sudoku.jpg`: Final image with the solved grid.

## Code Structure
- **ImageTransform**: A dataclass to store image transformation data (warped image, contour, etc.).
- **SudokuExtractor**: Main class handling:
  - Image preprocessing and contour detection.
  - Digit extraction using a pre-trained model.
  - Sudoku solving with a backtracking algorithm.
  - Visual output generation.
- **main()**: Entry point to parse arguments and run the solver.

## Notes
- The script assumes the input image contains a clear Sudoku grid with a roughly quadrilateral shape.
- The pre-trained model (`digit_models.h5`) must be trained on 28x28 grayscale digit images (similar to MNIST).
- If the Sudoku grid cannot be detected or solved, the script will print an error message and save diagnostic images for debugging.
- The confidence threshold for digit recognition is set to 0.7 to reduce false positives.

## Future Improvements
- Add support for multiple image formats.
- Improve digit recognition accuracy with advanced preprocessing.
- Implement a GUI for easier interaction.
- Optimize the backtracking algorithm for faster solving.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
