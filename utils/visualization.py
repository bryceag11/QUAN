# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry
import torchvision.transforms.functional as F
from metrics import obb_to_polygon, quaternion_to_angle
def plot_obb_on_image(image, obbs, categories, class_names, save_path):
    """
    Plot OBBs on the image.
    
    Args:
        image (torch.Tensor): Image tensor, shape (4, H, W).
        obbs (torch.Tensor): Bounding boxes, shape (N, 8) [x, y, w, h, qx, qy, qz, qw].
        categories (torch.Tensor): Category labels, shape (N,).
        class_names (dict): Mapping from class index to class name.
        save_path (str): Path to save the visualization.
    """
    # Convert quaternion to angle
    angles = quaternion_to_angle(obbs[:, 4:8]).cpu().numpy()
    
    # Convert OBBs to polygons
    polygons = [obb_to_polygon(obb.cpu().numpy()) for obb in obbs]
    
    # Convert image tensor to numpy array
    image_np = image[:3].cpu().numpy().transpose(1, 2, 0)  # Assuming first 3 channels are RGB
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    
    for poly, category, angle in zip(polygons, categories.cpu().numpy(), angles):
        x, y = poly.exterior.xy
        plt.plot(x, y, label=f"{class_names.get(category, 'N/A')} {angle:.2f} rad")
    
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, dpi=250)
    plt.close()
