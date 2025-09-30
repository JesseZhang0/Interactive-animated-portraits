import numpy as np
import random
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

def add_shape(img, shape_type="circle", position=None, size=None, color=None):
    """
    Overlays a shape onto the given img.
    
    Args:
        img (PIL.Image): The base img.
        shape_type (str): Type of shape to draw ("circle", "rectangle", etc.).
        position (tuple): Coordinates for the shape (e.g., (x, y)).
        size (int or tuple): Size of the shape.
        color (tuple): RGB color for the shape.
    
    Returns:
        PIL.Image: A copy of the img with the shape added.
    """
    new_image = img.copy()
    draw = ImageDraw.Draw(new_image)

    # Placeholder logic
    if shape_type == "circle":
        x, y = position if position else (random.randint(0, img.width), random.randint(0, img.height))
        r = size if size else random.randint(5, 50)
        color = color if color else (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)

    return new_image


def calculate_difference(image1, image2, diff_type="ssim"):
    """
    Computes a difference metric between two images.
    diff
    
    Args:
        image1, image2 (PIL.Image): Images to compare.
        diff_type (str): Metric to use ("ssim", "mse", "mae").
    
    Returns:
        float: A score representing difference (lower = more similar).
    """
    # Convert to numpy arrays
    arr1 = np.array(image1).astype(np.float32)
    arr2 = np.array(image2).astype(np.float32)
    
    # Ensure same size
    if arr1.shape != arr2.shape:
        raise ValueError("Images must be the same size for comparison.")
    
    # Compute difference
    diff = 0.0
    if diff_type == "ssim":
        score, _ = ssim(arr1, arr2, channel_axis=-1, full=True, data_range=arr2.max() - arr2.min())
        diff = 1 - score  # Higher SSIM means more similar, so invert it
    elif diff_type == "mse":
        diff = np.mean((arr1 - arr2) ** 2)
    elif diff_type == "mae":
        diff = np.mean(np.abs(arr1 - arr2))
    else:
        raise ValueError(f"Unknown diff_type: {diff_type}")
    
    return diff


def iterate(target_img, base_img=None, n=1, m=100, shape_type="circle", diff_type="mse"):
    """
    Iteratively improves base_img by adding shapes.
    
    Args:
        base_img (PIL.Image): Starting img.
        target_img (PIL.Image): Goal img to approximate.
        n (int): Number of iterations.
        m (int): Number of candidate shapes per iteration.
        shape_type (str): Type of shape to add.
        diff_type (str): Metric for image difference ("ssim", "mse", "mae").
    
    Returns:
        PIL.Image: The improved img.
    """
    # Initialize base image if not provided
    if base_img is None:
        # initialize as RGBA or RGB white background
        if target_img.mode == "RGBA":
            base_img = Image.new("RGBA", target_img.size, (255, 255, 255, 255))  # white background
        else:
            base_img = Image.new("RGB", target_img.size, (255, 255, 255))
    current_image = base_img.copy()

    for i in range(n):
        best_candidate = None
        best_score = calculate_difference(current_image, target_img, diff_type=diff_type)

        for j in range(m):
            candidate = add_shape(current_image, shape_type=shape_type)
            score = calculate_difference(candidate, target_img, diff_type=diff_type)

            if score < best_score:
                best_score = score
                best_candidate = candidate

        # Update with best candidate of this round
        if best_candidate:
            current_image = best_candidate
        else:
            print("No improvement found in this iteration.")
        print(f"Iteration {i+1}/{n}, Best Score: {best_score:.4f}")

    
    return current_image


if __name__ == "__main__":
    # Example usage: replace target.png with your target image path
    target = Image.open("Tech-tower-wreck.jpg").convert("RGBA")

    result = iterate(target, n=100, m=100)
    result.show()