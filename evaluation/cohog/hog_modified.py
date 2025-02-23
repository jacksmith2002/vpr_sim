import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit

stride = 2

@jit(nopython=True)
def get_closest_bins(gradient_angle, bin_size, angle_unit):
    idx = int(gradient_angle / angle_unit)
    mod = gradient_angle % angle_unit
    if idx == bin_size:
        return idx - 1, 0, mod
    return idx, (idx + 1) % bin_size, mod

@jit(nopython=True)
def cell_gradient(cell_magnitude, cell_angle, bin_size, angle_unit):
    """
    Compute the histogram for one cell, distributing gradient magnitudes
    into 2 adjacent bins (like classical HOG).
    """
    orientation_centers = np.zeros(bin_size, dtype=np.float64)
    h, w = cell_magnitude.shape

    for i in range(h):
        for j in range(w):
            strength = cell_magnitude[i, j]
            angle_val = cell_angle[i, j]

            min_angle, max_angle, mod = get_closest_bins(angle_val, bin_size, angle_unit)

            # Linear interpolation between the two nearest bins
            orientation_centers[min_angle] += strength * (1.0 - (mod / angle_unit))
            orientation_centers[max_angle] += strength * (mod / angle_unit)

    return orientation_centers

@jit(nopython=True)
def global_gradient(img):
    """
    Compute global gradient (Sobel) and return magnitude, angle.
    Numba cannot directly call cv2.Sobel in nopython mode, so
    you may need @jit(forceobj=True) or rewrite Sobel in NumPy.

    For demonstration, we do a simple approximation:
    """
    # If you'd like actual Sobel from OpenCV with Numba, you'll need object mode.
    # We'll do a naive gradient in pure NumPy to stay in nopython mode:
    # Gx ~ difference along x-axis, Gy ~ difference along y-axis
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)

    # A simple 3x3 Sobel-like approach:
    # For each pixel (1:-1, 1:-1), approximate partial derivatives
    h, w = img.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Gx kernel
            gx[y, x] = ( img[y - 1, x + 1] + 2*img[y, x + 1] + img[y + 1, x + 1]
                        - img[y - 1, x - 1] - 2*img[y, x - 1] - img[y + 1, x - 1] )
            # Gy kernel
            gy[y, x] = ( img[y + 1, x - 1] + 2*img[y + 1, x] + img[y + 1, x + 1]
                        - img[y - 1, x - 1] - 2*img[y - 1, x] - img[y - 1, x + 1] )

    magnitude = 0.5 * gx + 0.5 * gy  # somewhat imitating addWeighted(0.5, 0.5)
    angle = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            # angle in [0,360)
            angle[y, x] = math.degrees(math.atan2(gy[y, x], gx[y, x])) % 360.0

    return magnitude, angle

@jit(nopython=True)
def extract_hog(img, cell_size, bin_size, stride):
    """
    Numba-friendly HOG extraction:
      - No lambdas or generator expressions
      - Uses integer indexing
      - Returns a list of block descriptors
    """
    
    float_img = img.astype(np.float64)  # Ensure image is always float64

    height, width = float_img.shape

    # Convert to float64 and normalize
    max_val = np.max(float_img)
    if max_val > 0.0:
        norm_img = np.sqrt(float_img / max_val) * 255.0
    else:
        norm_img = float_img.copy()  # Avoid divide by zero

    # Compute gradients (pure NumPy or fallback to object mode if using cv2)
    grad_mag, grad_angle = global_gradient(norm_img)

    # Make them positive
    grad_mag = np.abs(grad_mag)

    angle_unit = 360 // bin_size

    # Dimensions in terms of cells
    cells_y = (height - cell_size) // stride + 1
    cells_x = (width - cell_size) // stride + 1

    # Accumulate histograms for each cell
    cell_gradient_vector = np.zeros((cells_y, cells_x, bin_size), dtype=np.float64)

    for i in range(cells_y):
        for j in range(cells_x):
            # slice out the cell
            row_start = i * stride
            col_start = j * stride
            cell_mag   = grad_mag[row_start : row_start + cell_size,
                                  col_start : col_start + cell_size]
            cell_angle = grad_angle[row_start : row_start + cell_size,
                                    col_start : col_start + cell_size]

            # compute per-cell histogram
            cell_hist = cell_gradient(cell_mag, cell_angle, bin_size, angle_unit)
            cell_gradient_vector[i, j, :] = cell_hist

    # Build the final HOG descriptor by grouping 2x2 cells into blocks
    # Each block => 4 cells => 4*bin_size dimension
    block_dim = bin_size * 4
    num_blocks_y = cells_y - 1
    num_blocks_x = cells_x - 1
    total_blocks = num_blocks_y * num_blocks_x

    # We'll store them in a 2D array: [total_blocks, block_dim]
    hog_descriptors = np.zeros((total_blocks, block_dim), dtype=np.float64)

    idx = 0
    for i in range(num_blocks_y):
        for j in range(num_blocks_x):
            # Concatenate 4 cells: (i,j), (i, j+1), (i+1,j), (i+1,j+1)
            block = np.zeros(block_dim, dtype=np.float64)
            # The easiest way is to flatten each cell's histogram and place them
            offset = 0
            for dy in range(2):
                for dx in range(2):
                    block[offset : offset + bin_size] = cell_gradient_vector[i + dy, j + dx]
                    offset += bin_size

            # L2 Normalize
            sq_sum = 0.0
            for val in block:
                sq_sum += val * val
            magnitude = math.sqrt(sq_sum)
            if magnitude > 0.0:
                for k in range(block_dim):
                    block[k] /= magnitude

            hog_descriptors[idx, :] = block
            idx += 1

    return hog_descriptors

class Hog_descriptor:
    """
    Thin wrapper class around the nopython-compiled 'extract_hog' function.
    """
    def __init__(self, img, cell_size=16, bin_size=9):
        self.img = img
        self.cell_size = cell_size
        self.bin_size = bin_size
        self.stride = 2  # from global stride
        # Basic assertion checks
        assert isinstance(self.cell_size, int), "cell_size must be int"
        assert isinstance(self.bin_size, int),  "bin_size must be int"

    def extract(self):
        return extract_hog(self.img, self.cell_size, self.bin_size, self.stride)

    def render_gradient(self, image, cell_gradient):
        """
        Optional debug function (not nopython) to visualize HOG arrows.
        Because it uses OpenCV drawing & Python loops, we keep it normal Python.
        """
        cell_width = self.cell_size / 2
        max_mag = np.array(cell_gradient).max()
        h, w = cell_gradient.shape[:2]
        for x in range(h):
            for y in range(w):
                cell_grad = cell_gradient[x, y]
                cell_grad /= max_mag
                angle_unit = 360 // self.bin_size
                angle = 0
                for magnitude in cell_grad:
                    angle_radian = math.radians(angle)
                    x1 = int(x * self.cell_size + magnitude * cell_width * math.cos(angle_radian))
                    y1 = int(y * self.cell_size + magnitude * cell_width * math.sin(angle_radian))
                    x2 = int(x * self.cell_size - magnitude * cell_width * math.cos(angle_radian))
                    y2 = int(y * self.cell_size - magnitude * cell_width * math.sin(angle_radian))
                    cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                    angle += angle_unit
        return image
