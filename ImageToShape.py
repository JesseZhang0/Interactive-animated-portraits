import numpy as np
import random
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

def add_shape(img, target_img=None, shape_type=None, position=None, size=None, color=None, scale_factor=1.0):
    """
    Overlays a random shape onto the given image.
    If target_img is provided, the shape color is the average color under the shape area.
    
    Args:
        img (PIL.Image): Image to draw on.
        target_img (PIL.Image): Reference image for color sampling.
        shape_type (str): 'circle', 'rectangle', or 'triangle'. If None, chosen randomly.
        position (tuple): (x, y) center of shape.
        size (int or tuple): size of the shape (radius or width/height).
        color (tuple): manual RGBA color, overrides sampling.
    """
    new_image = img.copy()
    draw = ImageDraw.Draw(new_image, "RGBA")

    # Pick random shape if not specified
    if shape_type is None:
        shape_type = np.random.choice(["circle", "rectangle", "triangle"])

    w, h = img.size
    x, y = position if position else (np.random.randint(0, w), np.random.randint(0, h))

    # Normalize size
    # --- Improved size randomization ---
    if size is None:
        # Make shape sizes scale with image dimensions for variety
        max_dim = min(w, h)
        base_scale = np.random.uniform(0.1, 1) * scale_factor # 10% to 100% of image dimension
        size_x = int(max_dim * base_scale)
        size_y = int(max_dim * np.random.uniform(0.1, 1) * scale_factor)
    elif isinstance(size, (int, float)):
        size_x = size_y = int(size)
    else:
        size_x, size_y = map(int, size)
    

    # Compute bounding box
    left, top = max(0, x - size_x), max(0, y - size_y)
    right, bottom = min(w, x + size_x), min(h, y + size_y)

    # Color sampling from target image
    if color is None and target_img is not None:
        region = target_img.crop((left, top, right, bottom)).convert("RGBA")
        region_np = np.array(region, dtype=np.float32)
        mask = np.zeros((bottom - top, right - left), dtype=bool)

        yy, xx = np.mgrid[top:bottom, left:right]
        cx, cy = x - left, y - top

        if shape_type == "circle":
            r = min(size_x, size_y)
            mask = (xx - left - cx) ** 2 + (yy - top - cy) ** 2 <= r ** 2

        elif shape_type == "rectangle":
            mask[:, :] = True

        elif shape_type == "triangle":
            # Define an upright triangle centered at (cx, cy)
            tri_height = size_y * 2
            tri_points = np.array([
                [cx, cy - tri_height / 2],
                [cx - size_x, cy + tri_height / 2],
                [cx + size_x, cy + tri_height / 2]
            ])
            # Use a polygon mask
            from matplotlib.path import Path
            grid_points = np.stack([xx - left, yy - top], axis=-1).reshape(-1, 2)
            mask_flat = Path(tri_points).contains_points(grid_points)
            mask = mask_flat.reshape(mask.shape)

        else:
            raise ValueError(f"Unsupported shape_type: {shape_type}")

        # Average color under mask
        if np.any(mask):
            masked_pixels = region_np[mask]
            avg_color = np.mean(masked_pixels, axis=0)
            color = tuple(avg_color.astype(np.uint8))
        else:
            color = (255, 255, 255, 255)

    elif color is None:
        # fallback random color
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            255
        )

    # Draw the shape
    if shape_type == "circle":
        draw.ellipse((left, top, right, bottom), fill=tuple(color))
    elif shape_type == "rectangle":
        draw.rectangle((left, top, right, bottom), fill=tuple(color))
    elif shape_type == "triangle":
        tri_points = [
            (x, y - size_y),
            (x - size_x, y + size_y),
            (x + size_x, y + size_y)
        ]
        draw.polygon(tri_points, fill=tuple(color))

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


def iterate(target_img, base_img=None, n=1, m=100, shape_type=None, diff_type="mse", scale_factor=1.0):
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

            candidate = add_shape(current_image, target_img, shape_type=shape_type, scale_factor=scale_factor)
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
    # result = iterate(target, base_img=result, n=500, m=50, scale_factor=0.5)
    result.show()