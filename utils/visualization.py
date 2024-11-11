# utils/visualization.py

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from matplotlib.patches import Polygon
import os
from typing import List, Tuple, Union
from .metrics import obb_to_polygon, quaternion_to_angle
from .ops import make_anchors, dist2bbox, dist2rbox

def process_predictions(predictions: dict, anchors: torch.Tensor, stride_tensor: torch.Tensor) -> dict:
    """
    Process raw predictions using anchors to get final boxes.
    
    Args:
        predictions (dict): Raw model predictions
        anchors (torch.Tensor): Anchor points
        stride_tensor (torch.Tensor): Stride tensor for each anchor
        
    Returns:
        dict: Processed predictions with decoded boxes
    """
    # Unpack predictions
    pred_distri = predictions.get('pred_dist')  # Distribution predictions
    pred_scores = predictions.get('pred_scores')  # Classification scores
    pred_quats = predictions.get('pred_quats')  # Quaternion predictions

    # Process boxes using anchors
    if pred_distri is not None:
        # Convert distributions to boxes
        pred_boxes = dist2rbox(anchors, pred_distri.view(-1, 4))  # Shape: (N, 4)
        pred_boxes = pred_boxes * stride_tensor.view(-1, 1)
    else:
        pred_boxes = predictions.get('pred_boxes')  # Use direct box predictions if available

    return {
        'boxes': pred_boxes,
        'scores': pred_scores,
        'quats': pred_quats,
        'labels': predictions.get('pred_labels')
    }

def quaternion_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    quats: torch.Tensor,
    iou_threshold: float = 0.5
) -> torch.Tensor:
    """
    Non-Maximum Suppression for OBBs with quaternions.
    
    Args:
        boxes (torch.Tensor): Boxes in (x, y, w, h) format, shape (N, 4)
        scores (torch.Tensor): Box scores, shape (N,)
        quats (torch.Tensor): Quaternions, shape (N, 4)
        iou_threshold (float): IoU threshold for NMS
    
    Returns:
        torch.Tensor: Indices of kept boxes
    """
    if boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Convert boxes and quaternions to polygons for IoU computation
    polygons = []
    for box, quat in zip(boxes, quats):
        polygon = obb_to_polygon(torch.cat([box, quat]))
        polygons.append(np.array(polygon.exterior.coords)[:-1])

    # Compute areas
    areas = torch.tensor([cv2.contourArea(poly) for poly in polygons],
                        device=boxes.device)

    # Sort boxes by score
    _, order = scores.sort(0, descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0]
        keep.append(i)

        # Compute IoU of the current box with rest
        ious = torch.zeros(len(order) - 1, device=boxes.device)
        for j, idx in enumerate(order[1:]):
            poly1 = polygons[i].astype(np.float32)
            poly2 = polygons[idx].astype(np.float32)
            inter = cv2.intersectConvexConvex(poly1, poly2)[0]
            if inter is not None:
                inter_area = cv2.contourArea(inter)
                iou = inter_area / (areas[i] + areas[idx] - inter_area)
                ious[j] = iou

        ids = (ious <= iou_threshold).nonzero().view(-1)
        if ids.numel() == 0:
            break
        order = order[ids + 1]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

def plot_batch_predictions(
    images: torch.Tensor,
    predictions: dict,
    targets: dict,
    anchors: torch.Tensor,
    stride_tensor: torch.Tensor,
    class_names: dict,
    save_dir: str,
    batch_idx: int,
    max_images: int = 4,
    plot_anchors: bool = False
) -> None:
    """
    Plot predictions and ground truth for a batch of images.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W)
        predictions (dict): Model predictions containing boxes, scores, quats, labels
        targets (dict): Ground truth annotations
        anchors (torch.Tensor): Anchor points
        stride_tensor (torch.Tensor): Stride tensor for each anchor
        class_names (dict): Mapping from label to class name
        save_dir (str): Directory to save visualizations
        batch_idx (int): Batch index for filename
        max_images (int): Maximum number of images to plot from batch
        plot_anchors (bool): Whether to plot anchor points
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Process predictions with anchors
    processed_preds = process_predictions(predictions, anchors, stride_tensor)
    
    # Plot only up to max_images
    num_images = min(len(images), max_images)
    
    for i in range(num_images):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Convert image from tensor to numpy
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        if img.shape[0] == 4:  # If quaternion image
            img = img[:3]  # Take only RGB channels
        
        # Plot predictions
        ax1.imshow(img)
        ax1.set_title('Predictions')
        
        # Plot anchor points if requested
        if plot_anchors:
            anchor_points = anchors.cpu().numpy()
            ax1.scatter(anchor_points[:, 0], anchor_points[:, 1], c='blue', alpha=0.3, s=1)
        
        if processed_preds is not None:
            boxes = processed_preds['boxes'][i]
            scores = processed_preds['scores'][i]
            quats = processed_preds['quats'][i]
            labels = processed_preds['labels'][i]
            
            # Apply NMS
            keep = quaternion_nms(boxes, scores, quats)
            
            # Plot kept boxes
            for box, quat, label, score in zip(boxes[keep], quats[keep], labels[keep], scores[keep]):
                polygon = obb_to_polygon(torch.cat([box, quat]))
                x, y = polygon.exterior.xy
                ax1.plot(x, y, 'r-', linewidth=2)
                class_name = class_names.get(label.item(), 'unknown')
                ax1.text(min(x), min(y), f'{class_name}\n{score:.2f}',
                        bbox=dict(facecolor='white', alpha=0.7))
        
        # Plot ground truth
        ax2.imshow(img)
        ax2.set_title('Ground Truth')
        if targets is not None:
            gt_boxes = targets['bbox'][i]
            gt_quats = targets['quat'][i]
            gt_labels = targets['category'][i]
            
            for box, quat, label in zip(gt_boxes, gt_quats, gt_labels):
                polygon = obb_to_polygon(torch.cat([box, quat]))
                x, y = polygon.exterior.xy
                ax2.plot(x, y, 'g-', linewidth=2)
                class_name = class_names.get(label.item(), 'unknown')
                ax2.text(min(x), min(y), class_name,
                        bbox=dict(facecolor='white', alpha=0.7))
        
        plt.savefig(os.path.join(save_dir, f'batch_{batch_idx}_img_{i}.png'))
        plt.close(fig)

def visualize_training_batch(
    batch: dict,
    predictions: dict,
    anchors: torch.Tensor,
    stride_tensor: torch.Tensor,
    class_names: dict,
    save_dir: str,
    batch_idx: int,
    epoch: int,
    plot_anchors: bool = False
) -> None:
    """
    Visualize a training batch with predictions and ground truth.
    
    Args:
        batch (dict): Training batch containing images and targets
        predictions (dict): Model predictions
        anchors (torch.Tensor): Anchor points
        stride_tensor (torch.Tensor): Stride tensor for each anchor
        class_names (dict): Class name mapping
        save_dir (str): Directory to save visualizations
        batch_idx (int): Batch index
        epoch (int): Current epoch
        plot_anchors (bool): Whether to plot anchor points
    """
    vis_dir = os.path.join(save_dir, f'epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    plot_batch_predictions(
        images=batch['image'],
        predictions=predictions,
        targets={
            'bbox': batch['bbox'],
            'quat': batch['quat'],
            'category': batch['category']
        },
        anchors=anchors,
        stride_tensor=stride_tensor,
        class_names=class_names,
        save_dir=vis_dir,
        batch_idx=batch_idx,
        plot_anchors=plot_anchors
    )

def plot_obb_on_image(image, obbs, categories, class_names, save_path, anchors=None):
    """
    Plot OBBs on the image.
    
    Args:
        image (torch.Tensor): Image tensor, shape (4, H, W)
        obbs (torch.Tensor): Bounding boxes, shape (N, 8) [x, y, w, h, qx, qy, qz, qw]
        categories (torch.Tensor): Category labels, shape (N,)
        class_names (dict): Mapping from class index to class name
        save_path (str): Path to save the visualization
        anchors (torch.Tensor, optional): Anchor points to plot
    """
    angles = quaternion_to_angle(obbs[:, 4:8]).cpu().numpy()
    polygons = [obb_to_polygon(obb.cpu().numpy()) for obb in obbs]
    image_np = image[:3].cpu().numpy().transpose(1, 2, 0)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image_np)
    
    # Plot anchor points if provided
    if anchors is not None:
        anchor_points = anchors.cpu().numpy()
        plt.scatter(anchor_points[:, 0], anchor_points[:, 1], c='blue', alpha=0.3, s=1)
    
    for poly, category, angle in zip(polygons, categories.cpu().numpy(), angles):
        x, y = poly.exterior.xy
        plt.plot(x, y, label=f"{class_names.get(category, 'N/A')} {angle:.2f} rad")
    
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path, dpi=250)
    plt.close()