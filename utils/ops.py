# utils/quaternion_ops.py

import torch
from itertools import product
import seaborn
import numpy as np 
import shapely.geometry
import math
  

def bbox2dist(anchor_points, bboxes, reg_max=16, xywh=True):
    """
    Convert bounding box coordinates to distance distributions for training.
    
    Args:
        anchor_points (torch.Tensor): Anchor points, shape (N_anchors, 2) or (B, N_anchors, 2)
        bboxes (torch.Tensor): Bounding boxes, shape (N_boxes, 4) or (B, N_boxes, 4)
        reg_max (int): Maximum value for distance bins
        xywh (bool): If True, boxes are in [x, y, w, h] format
    """
    # Handle batched inputs
    is_batched = bboxes.dim() == 3
    if not is_batched:
        bboxes = bboxes.unsqueeze(0)
        anchor_points = anchor_points.unsqueeze(0)
    
    B = bboxes.shape[0]
    N_boxes = bboxes.shape[1]  # Number of target boxes
    N_anchors = anchor_points.shape[1]  # Number of anchor points
    
    # Convert boxes to xyxy if needed
    if xywh:
        x_center, y_center, w, h = bboxes.unbind(-1)
        x1 = x_center - w/2
        y1 = y_center - h/2
        x2 = x_center + w/2
        y2 = y_center + h/2
    else:
        x1, y1, x2, y2 = bboxes.unbind(-1)
    
    device = bboxes.device
    
    # Initialize output distributions
    dist = torch.zeros((B, N_anchors, 4 * reg_max), device=device)
    
    # Process each batch
    for b in range(B):
        # Get anchors and coordinates for this batch
        batch_anchors = anchor_points[b]  # [N_anchors, 2]
        anchor_x, anchor_y = batch_anchors.unbind(-1)  # [N_anchors]
        
        # Compute distances to nearest boxes
        anchor_points_expanded = batch_anchors.unsqueeze(1)  # [N_anchors, 1, 2]
        box_centers = torch.stack([
            (x1[b] + x2[b])/2,
            (y1[b] + y2[b])/2
        ], dim=-1)  # [N_boxes, 2]
        
        # Find nearest boxes
        distances = torch.norm(
            anchor_points_expanded - box_centers.unsqueeze(0),
            dim=-1
        )  # [N_anchors, N_boxes]
        box_indices = distances.argmin(dim=1)  # [N_anchors]
        
        # Calculate distances to box edges
        d_left = torch.clamp(
            anchor_x - x1[b][box_indices],
            min=0, max=reg_max-1
        ).long()  # [N_anchors]
        
        d_top = torch.clamp(
            anchor_y - y1[b][box_indices],
            min=0, max=reg_max-1
        ).long()  # [N_anchors]
        
        d_right = torch.clamp(
            x2[b][box_indices] - anchor_x,
            min=0, max=reg_max-1
        ).long()  # [N_anchors]
        
        d_bottom = torch.clamp(
            y2[b][box_indices] - anchor_y,
            min=0, max=reg_max-1
        ).long()  # [N_anchors]
        
        # Fill in one-hot encodings
        # Use offset for each coordinate's bins
        for i, d_idx in enumerate([d_left, d_top, d_right, d_bottom]):
            offset = i * reg_max
            for anchor_idx, bin_idx in enumerate(d_idx):
                dist[b, anchor_idx, offset + bin_idx] = 1.0
    
    # Remove batch dimension if input wasn't batched
    if not is_batched:
        dist = dist.squeeze(0)
    
    return dist

def dist2bbox(distances, anchor_points, xywh=True):
    """
    Convert distance predictions to bounding boxes.
    
    Args:
        distances (torch.Tensor): Distance predictions
            Shape can be either:
            - (N, 4) for unbatched
            - (B, N, 4) for batched
        anchor_points (torch.Tensor): Anchor points, shape (N, 2)
        xywh (bool): If True, return boxes in (x,y,w,h) format, else in (x1,y1,x2,y2)
    
    Returns:
        torch.Tensor: Bounding boxes in same batch format as input
    """
    # Check if input is batched
    is_batched = distances.dim() == 3
    if not is_batched:
        distances = distances.unsqueeze(0)  # Add batch dim
        
    batch_size = distances.size(0)
    num_anchors = anchor_points.size(0)
    
    # Expand anchor points for batch processing
    # [N, 2] -> [1, N, 2] -> [B, N, 2]
    anchor_points = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Get anchor coordinates, now with batch dimension
    # [B, N]
    anchor_x = anchor_points[..., 0]
    anchor_y = anchor_points[..., 1]
    
    if xywh:
        # center x, center y (maintain batch dimension)
        c_x = anchor_x + distances[..., 0]  # [B, N]
        c_y = anchor_y + distances[..., 1]  # [B, N]
        # width, height
        w = distances[..., 2].exp()  # [B, N]
        h = distances[..., 3].exp()  # [B, N]
        
        boxes = torch.stack([c_x, c_y, w, h], dim=-1)  # [B, N, 4]
    else:
        # Convert to xyxy format
        x1 = anchor_x - distances[..., 0]  # [B, N]
        y1 = anchor_y - distances[..., 1]  # [B, N]
        x2 = anchor_x + distances[..., 2]  # [B, N]
        y2 = anchor_y + distances[..., 3]  # [B, N]
        
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, N, 4]
    
    # Remove batch dimension if input was unbatched
    if not is_batched:
        boxes = boxes.squeeze(0)
        
    return boxes

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