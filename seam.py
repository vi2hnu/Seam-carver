import numpy as np
from PIL import Image
import cv2

def calculate_energy_map(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    energy_map = np.sqrt(sobel_x**2 + sobel_y**2)
    energy_map = (energy_map / np.max(energy_map) * 255).astype(np.uint8)
    return energy_map

def compute_cumulative_energy(energy_map):
    rows, cols = energy_map.shape
    cumulative_energy = np.copy(energy_map).astype(np.float64)
    
    for i in range(1, rows):
        left = np.roll(cumulative_energy[i-1, :], shift=1)
        left[0] = np.inf
        right = np.roll(cumulative_energy[i-1, :], shift=-1)
        right[-1] = np.inf
        
        cumulative_energy[i, :] += np.minimum(np.minimum(left, cumulative_energy[i-1, :]), right)
    
    return cumulative_energy

def find_min_seam(cumulative_energy):
    rows, cols = cumulative_energy.shape
    seam = np.zeros(rows, dtype=np.int64)
    
    seam[-1] = np.argmin(cumulative_energy[-1])
    
    for i in range(rows - 2, -1, -1):
        j = seam[i + 1]
        if j > 0 and cumulative_energy[i, j - 1] == min(cumulative_energy[i, max(j - 1, 0):min(j + 2, cols)]):
            seam[i] = j - 1
        elif j < cols - 1 and cumulative_energy[i, j + 1] == min(cumulative_energy[i, max(j - 1, 0):min(j + 2, cols)]):
            seam[i] = j + 1
        else:
            seam[i] = j
    
    return seam

def remove_seam(image, seam):
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols - 1, 3), dtype=np.uint8)
    
    for row in range(rows):
        col = seam[row]
        new_image[row, :, :] = np.concatenate([
            image[row, :col, :],
            image[row, col + 1:, :]
        ], axis=0)
    
    return new_image

#additon of seam or expanding the image is not work properly, needs improvement
def add_seam(image, seam):
    rows, cols, _ = image.shape
    new_image = np.zeros((rows, cols + 1, 3), dtype=np.uint8)
    
    for row in range(rows):
        col = seam[row]
        new_image[row, :col, :] = image[row, :col, :]
        new_image[row, col, :] = image[row, col, :]
        new_image[row, col + 1:, :] = image[row, col:, :]
    return new_image

# Load the image
src = np.array(Image.open("image.jpg"))
# Carve the image
for _ in range(300):
    energy_map = calculate_energy_map(src)
    cumulative_energy = compute_cumulative_energy(energy_map)
    seam = find_min_seam(cumulative_energy)
    src = remove_seam(src, seam)

# Convert the carved image to PIL Image and display it
# Convert the carved image to PIL Image
carved_image_pil = Image.fromarray(src)

# Save the carved image to a file
carved_image_pil.save("carved_image.jpg")

