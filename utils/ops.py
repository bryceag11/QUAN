# utils/quaternion_ops.py

import torch
from itertools import product
import seaborn
import numpy as np 
import shapely.geometry
import math

def bbox2dist(anchor_points, target_bboxes, reg_max):
    """
    Convert bounding boxes to distances for DFL.
    
    Args:
        anchor_points: [num_anchors, 2]
        target_bboxes: [num_anchors, 4]
        reg_max: int
    
    Returns:
        distances: [num_anchors, 4]
    """
    x1, y1, x2, y2 = target_bboxes[:, 0], target_bboxes[:, 1], target_bboxes[:, 2], target_bboxes[:, 3]
    xa, ya = anchor_points[:, 0], anchor_points[:, 1]
    
    # Calculate distances
    d_l = xa - x1
    d_t = ya - y1
    d_r = x2 - xa
    d_b = y2 - ya
    
    # Clamp distances to [0, reg_max]
    d_l = torch.clamp(d_l, min=0, max=reg_max)
    d_t = torch.clamp(d_t, min=0, max=reg_max)
    d_r = torch.clamp(d_r, min=0, max=reg_max)
    d_b = torch.clamp(d_b, min=0, max=reg_max)
    
    distances = torch.stack([d_l, d_t, d_r, d_b], dim=1)
    return distances


def dist2bbox(distances, anchor_points, xywh=True):
    """
    Convert distances to bounding boxes with clamping to ensure numerical stability.
    
    Args:
        distances: [batch_size, num_anchors, 4]
        anchor_points: [num_anchors, 2]
        xywh: bool, whether to return (x, y, w, h) or (x1, y1, x2, y2)
    
    Returns:
        bboxes: [batch_size, num_anchors, 4]
    """
    if xywh:
        x = anchor_points[:, 0].unsqueeze(0).expand_as(distances[:, :, 0])
        y = anchor_points[:, 1].unsqueeze(0).expand_as(distances[:, :, 1])
        w = distances[:, :, 0] + distances[:, :, 2]
        h = distances[:, :, 1] + distances[:, :, 3]
        
        # Clamp width and height to prevent negative or extreme values
        w = torch.clamp(w, min=1.0, max=1e4)
        h = torch.clamp(h, min=1.0, max=1e4)
        
        bboxes = torch.stack([x, y, w, h], dim=2)
    else:
        x1 = anchor_points[:, 0].unsqueeze(0).expand_as(distances[:, :, 0]) - distances[:, :, 0]
        y1 = anchor_points[:, 1].unsqueeze(0).expand_as(distances[:, :, 1]) - distances[:, :, 1]
        x2 = anchor_points[:, 0].unsqueeze(0).expand_as(distances[:, :, 2]) + distances[:, :, 2]
        y2 = anchor_points[:, 1].unsqueeze(0).expand_as(distances[:, :, 3]) + distances[:, :, 3]
        
        # Clamp coordinates to prevent negative or extreme values
        x1 = torch.clamp(x1, min=0.0, max=1e4)
        y1 = torch.clamp(y1, min=0.0, max=1e4)
        x2 = torch.clamp(x2, min=0.0, max=1e4)
        y2 = torch.clamp(y2, min=0.0, max=1e4)
        
        bboxes = torch.stack([x1, y1, x2, y2], dim=2)
    return bboxes


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features.
    
    Args:
        feats (list[torch.Tensor]): List of feature maps
        strides (torch.Tensor): Strides for each feature map
        grid_cell_offset (float): Offset for grid cells
        
    Returns:
        tuple: anchor_points, stride_tensor
    """
    anchor_points, stride_tensor = [], []
    
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(w, device=feats[i].device, dtype=torch.float32) + grid_cell_offset  # shift x
        sy = torch.arange(h, device=feats[i].device, dtype=torch.float32) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2) * stride)
        stride_tensor.append(torch.full((h * w,), stride, device=feats[i].device))
    
    return torch.cat(anchor_points), torch.cat(stride_tensor)


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

    if bboxes.numel() == 0:
        print("Warning: Empty bboxes tensor!")
        return

    if xywh:
        # Ensure bboxes has correct shape
        if bboxes.dim() != 2 or bboxes.shape[1] != 4:
            raise ValueError(f"Expected bboxes shape (N, 4), got {bboxes.shape}")
            
        x_center, y_center, width, height = bboxes.unbind(1)
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
    else:
        x1, y1, x2, y2 = bboxes.unbind(1)

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


def dist2rbox(anchor_points, pred_dist, xywh=True, dim=-1):
    """
    Convert distance predictions to rotated bounding boxes.

    Args:
        anchor_points (torch.Tensor): Anchor points, shape (N, 2) or (batch, N, 2)
        pred_dist (torch.Tensor): Distance predictions, shape (N, 4) or (batch, N, 4)
        xywh (bool): If True, return boxes in xywh format, else return in xyxy format
        dim (int): Dimension along which to split predictions

    Returns:
        torch.Tensor: Rotated bounding boxes in chosen format
    """
    # Ensure inputs have compatible shapes
    if pred_dist.dim() == 3 and anchor_points.dim() == 2:
        anchor_points = anchor_points.unsqueeze(0).expand(pred_dist.size(0), -1, -1)
    
    # Split predictions
    if pred_dist.size(dim) == 4:  # Standard distance predictions
        distance = pred_dist
    else:  # Predictions include angle
        distance, angle = torch.split(pred_dist, [4, 1], dim=dim)
    
    # Convert distances to box parameters
    if xywh:
        # Center coordinates
        c_x = anchor_points[..., 0] + distance[..., 0]
        c_y = anchor_points[..., 1] + distance[..., 1]
        # Width and height
        w = distance[..., 2].exp()
        h = distance[..., 3].exp()
        
        if distance.size(dim) > 4:  # If we have angle predictions
            # Add rotation parameters
            cos_a = torch.cos(angle[..., 0])
            sin_a = torch.sin(angle[..., 0])
            
            # Create rotated box coordinates
            x1 = c_x - w/2 * cos_a + h/2 * sin_a
            y1 = c_y - w/2 * sin_a - h/2 * cos_a
            x2 = c_x + w/2 * cos_a + h/2 * sin_a
            y2 = c_y + w/2 * sin_a - h/2 * cos_a
            
            return torch.stack((c_x, c_y, w, h, angle[..., 0]), dim=dim)
        else:
            return torch.stack((c_x, c_y, w, h), dim=dim)
    else:
        # Convert to xyxy format
        x1 = anchor_points[..., 0] + distance[..., 0]
        y1 = anchor_points[..., 1] + distance[..., 1]
        x2 = anchor_points[..., 0] + distance[..., 2]
        y2 = anchor_points[..., 1] + distance[..., 3]
        
        if distance.size(dim) > 4:  # If we have angle predictions
            return torch.stack((x1, y1, x2, y2, angle[..., 0]), dim=dim)
        else:
            return torch.stack((x1, y1, x2, y2), dim=dim)

def rbox2dist(anchor_points, rbox, reg_max):
    """
    Convert rotated bounding boxes to distance predictions.

    Args:
        anchor_points (torch.Tensor): Anchor points (N, 2)
        rbox (torch.Tensor): Rotated bounding boxes (N, 5) in [x, y, w, h, angle] format
        reg_max (int): Maximum value for distance bins

    Returns:
        torch.Tensor: Distance predictions and angle
    """
    # Extract box parameters
    x, y, w, h, angle = rbox.unbind(-1)
    
    # Calculate distances
    dist_x = x - anchor_points[..., 0]
    dist_y = y - anchor_points[..., 1]
    dist_w = w.log()
    dist_h = h.log()
    
    # Clip distances to reg_max
    dist = torch.stack((dist_x, dist_y, dist_w, dist_h), -1).clamp(-reg_max, reg_max)
    
    # Include angle
    angle = angle.unsqueeze(-1)
    return torch.cat([dist, angle], dim=-1)