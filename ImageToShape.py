import numpy as np
import random
from PIL import Image, ImageDraw

def add_shape(image, shape_type="circle", position=None, size=None, color=None):
    """
    Overlays a shape onto the given image.
    
    Args:
        image (PIL.Image): The base image.
        shape_type (str): Type of shape to draw ("circle", "rectangle", etc.).
        position (tuple): Coordinates for the shape (e.g., (x, y)).
        size (int or tuple): Size of the shape.
        color (tuple): RGB color for the shape.
    
    Returns:
        PIL.Image: A copy of the image with the shape added.
    """
    new_image = image.copy()
    draw = ImageDraw.Draw(new_image)

    # Placeholder logic
    if shape_type == "circle":
        x, y = position if position else (random.randint(0, image.width), random.randint(0, image.height))
        r = size if size else random.randint(5, 50)
        color = color if color else (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

    return new_image


def calculate_difference(image1, image2):
    """
    Computes a difference metric between two images.
    
    Args:
        image1, image2 (PIL.Image): Images to compare.
    
    Returns:
        float: A score representing difference (lower = more similar).
    """
    # Convert to numpy arrays
    arr1 = np.array(image1).astype(np.float32)
    arr2 = np.array(image2).astype(np.float32)
    
    # Example: Mean squared error
    diff = np.mean((arr1 - arr2) ** 2)
    return diff


def iterate(base_image, target_image, n=10, m=5):
    """
    Iteratively improves base_image by adding shapes.
    
    Args:
        base_image (PIL.Image): Starting image.
        target_image (PIL.Image): Goal image to approximate.
        n (int): Number of iterations.
        m (int): Number of candidate shapes per iteration.
    
    Returns:
        PIL.Image: The improved image.
    """
    current_image = base_image.copy()

    for i in range(n):
        best_candidate = None
        best_score = float("inf")

        for j in range(m):
            candidate = add_shape(current_image)
            score = calculate_difference(candidate, target_image)

            if score < best_score:
                best_score = score
                best_candidate = candidate

        # Update with best candidate of this round
        current_image = best_candidate
    
    return current_image


if __name__ == "__main__":
    # Example usage
    target = Image.new("RGB", (128, 128), (255, 255, 255))  # white target
    base = Image.new("RGB", (128, 128), (0, 0, 0))          # black base

    result = iterate(base, target, n=10, m=5)
    result.show()