# utils/quaternion_ops.py

import torch
from itertools import product
import seaborn
import numpy as np 
import shapely.geometry


def dist2bbox(anchor_points, pred_dist, xywh=True):
    """
    Convert distance predictions to bounding box coordinates.

    Args:
        anchor_points (torch.Tensor): Anchor points, shape (N, 2).
        pred_dist (torch.Tensor): Predicted distances, shape (N, 4 * reg_max).
        xywh (bool): If True, return [x, y, w, h], else [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Bounding boxes, shape (N, 4).
    """
    reg_max = pred_dist.shape[1] // 4
    pred_dist = pred_dist.view(-1, 4, reg_max).softmax(dim=2).matmul(torch.arange(reg_max, device=pred_dist.device).float().unsqueeze(0))
    pred_dist = pred_dist * (reg_max - 1)

    if xywh:
        x = anchor_points[:, 0] - pred_dist[:, 0]
        y = anchor_points[:, 1] - pred_dist[:, 1]
        w = pred_dist[:, 2] + pred_dist[:, 0]
        h = pred_dist[:, 3] + pred_dist[:, 1]
        return torch.stack([x, y, w, h], dim=1)
    else:
        x1 = anchor_points[:, 0] - pred_dist[:, 0]
        y1 = anchor_points[:, 1] - pred_dist[:, 1]
        x2 = anchor_points[:, 0] + pred_dist[:, 2]
        y2 = anchor_points[:, 1] + pred_dist[:, 3]
        return torch.stack([x1, y1, x2, y2], dim=1)

def dist2rbox(anchor_points, pred_dist, reg_max=16):
    """
    Convert distance predictions to rotated bounding box coordinates with quaternions.

    Args:
        anchor_points (torch.Tensor): Anchor points, shape (N, 2).
        pred_dist (torch.Tensor): Predicted distances, shape (N, 4 * reg_max).
        reg_max (int): Maximum value for distribution focal loss.

    Returns:
        torch.Tensor: Rotated bounding boxes, shape (N, 8) [x, y, w, h, qx, qy, qz, qw].
    """
    reg_max = pred_dist.shape[1] // 4
    pred_dist = pred_dist.view(-1, 4, reg_max).softmax(dim=2).matmul(torch.arange(reg_max, device=pred_dist.device).float().unsqueeze(0))
    pred_dist = pred_dist * (reg_max - 1)

    x = anchor_points[:, 0] - pred_dist[:, 0]
    y = anchor_points[:, 1] - pred_dist[:, 1]
    w = pred_dist[:, 2] + pred_dist[:, 0]
    h = pred_dist[:, 3] + pred_dist[:, 1]

    # Placeholder for quaternion. Replace with actual quaternion predictions.
    # Assuming you have separate quaternion predictions.
    quat = torch.zeros((pred_dist.shape[0], 4), device=pred_dist.device)
    quat[:, 3] = 1.0  # Initialize with no rotation

    return torch.cat([x.unsqueeze(1), y.unsqueeze(1), w.unsqueeze(1), h.unsqueeze(1), quat], dim=1)


def make_anchors(feats, strides, offset=0.5):
    """
    Generate anchor points for each feature map.

    Args:
        feats (list): List of feature map tensors.
        strides (list): List of strides for each feature map.
        offset (float): Offset to center the anchors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Anchor points and stride tensor.
    """
    anchor_points = []
    stride_tensor = []
    for i, feat in enumerate(feats):
        h, w = feat.shape[-2:]
        stride = strides[i]
        grid_y, grid_x = torch.meshgrid(torch.arange(h, device=feat.device), torch.arange(w, device=feat.device))
        grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).float()
        grid = grid * stride + stride * offset
        anchor_points.append(grid)
        stride_tensor.append(torch.full((grid.shape[0],), stride, device=feat.device))
    anchor_points = torch.cat(anchor_points, dim=0)
    stride_tensor = torch.cat(stride_tensor, dim=0)
    return anchor_points, stride_tensor

def xywh2xyxy(boxes):
    """
    Convert [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        boxes (torch.Tensor): Bounding boxes, shape (N, 4).

    Returns:
        torch.Tensor: Converted bounding boxes, shape (N, 4).
    """
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


def xyxy2xywh(boxes):
    """
    Convert [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        boxes (torch.Tensor): Bounding boxes, shape (N, 4).

    Returns:
        torch.Tensor: Converted bounding boxes, shape (N, 4).
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    w = x2 - x1
    h = y2 - y1
    x = x1 + w / 2
    y = y1 + h / 2
    return torch.stack([x, y, w, h], dim=1)


def crop_mask(mask, bbox):
    """
    Crop the mask based on the bounding box.

    Args:
        mask (torch.Tensor): Binary mask, shape (H, W).
        bbox (torch.Tensor): Bounding box, [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Cropped mask, shape (crop_h, crop_w).
    """
    x1, y1, x2, y2 = bbox.int()
    cropped = mask[y1:y2, x1:x2]
    return cropped

def bbox2dist(anchor_points, bboxes, reg_max=16, xywh=True):
    """
    Convert bounding box coordinates to distance distributions for training.

    Args:
        anchor_points (torch.Tensor): Anchor points, shape (N, 2).
        bboxes (torch.Tensor): Bounding boxes, shape (N, 4).
            - If xywh=True: [x_center, y_center, width, height]
            - If xywh=False: [x1, y1, x2, y2]
        reg_max (int, optional): Maximum value for distance bins. Default is 16.
        xywh (bool, optional): If True, boxes are in [x, y, w, h] format. Else, [x1, y1, x2, y2].

    Returns:
        torch.Tensor: Distance distributions, shape (N, 4 * reg_max).
    """
    if xywh:
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        x_center, y_center, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
    else:
        # Assume bboxes are already in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    # Extract anchor coordinates
    anchor_x, anchor_y = anchor_points[:, 0], anchor_points[:, 1]

    # Calculate distances from anchors to bbox sides
    d_left = anchor_x - x1      # Distance from anchor to left side of bbox
    d_top = anchor_y - y1       # Distance from anchor to top side of bbox
    d_right = x2 - anchor_x     # Distance from anchor to right side of bbox
    d_bottom = y2 - anchor_y    # Distance from anchor to bottom side of bbox

    # Clamp distances to [0, reg_max - 1] to avoid overflow
    d_left = torch.clamp(d_left, min=0, max=reg_max - 1)
    d_top = torch.clamp(d_top, min=0, max=reg_max - 1)
    d_right = torch.clamp(d_right, min=0, max=reg_max - 1)
    d_bottom = torch.clamp(d_bottom, min=0, max=reg_max - 1)

    # Convert distances to integer bin indices
    d_left_idx = d_left.long()
    d_top_idx = d_top.long()
    d_right_idx = d_right.long()
    d_bottom_idx = d_bottom.long()

    # Initialize one-hot encodings for each distance
    N = anchor_points.size(0)
    device = anchor_points.device

    d_left_onehot = torch.zeros((N, reg_max), device=device)
    d_top_onehot = torch.zeros((N, reg_max), device=device)
    d_right_onehot = torch.zeros((N, reg_max), device=device)
    d_bottom_onehot = torch.zeros((N, reg_max), device=device)

    # Scatter 1s at the bin indices
    d_left_onehot.scatter_(1, d_left_idx.unsqueeze(1), 1)
    d_top_onehot.scatter_(1, d_top_idx.unsqueeze(1), 1)
    d_right_onehot.scatter_(1, d_right_idx.unsqueeze(1), 1)
    d_bottom_onehot.scatter_(1, d_bottom_idx.unsqueeze(1), 1)

    # Concatenate all distance distributions
    dist = torch.cat([d_left_onehot, d_top_onehot, d_right_onehot, d_bottom_onehot], dim=1)  # Shape: (N, 4 * reg_max)

    return dist

def bbox_to_obb_no_rotation(bbox):
    """
    Convert [x, y, w, h] to [x_center, y_center, w, h, qx, qy, qz, qw]
    with quaternion representing no rotation.
    
    Args:
        bbox (list or torch.Tensor): [x, y, w, h]
    
    Returns:
        list: [x_center, y_center, w, h, 0, 0, 0, 1]
    """
    x, y, w, h = bbox
    x_center = x + w / 2
    y_center = y + h / 2
    return [x_center, y_center, w, h, 0, 0, 0, 1]

def polygon_to_obb(polygon):
    """
    Convert a polygon to an Oriented Bounding Box (OBB) with quaternion.

    Args:
        polygon (list or np.ndarray): List of coordinates [x1, y1, x2, y2, x3, y3, x4, y4].

    Returns:
        list: [x, y, w, h, qx, qy, qz, qw]
    """
    # Create a Shapely polygon
    poly = shapely.geometry.Polygon(polygon).minimum_rotated_rectangle
    coords = np.array(poly.exterior.coords)[:-1]  # Remove duplicate last point

    # Calculate center
    x = coords[:, 0].mean()
    y = coords[:, 1].mean()

    # Calculate width and height
    edge_lengths = np.linalg.norm(coords - np.roll(coords, -1, axis=0), axis=1)
    w, h = sorted(edge_lengths)[:2]

    # Calculate angle
    edge = coords[1] - coords[0]
    theta = math.atan2(edge[1], edge[0])  # Rotation angle in radians

    # Convert angle to quaternion (assuming rotation around z-axis)
    half_angle = theta / 2.0
    qx = 0.0
    qy = 0.0
    qz = math.sin(half_angle)
    qw = math.cos(half_angle)

    return [x, y, w, h, qx, qy, qz, qw]

