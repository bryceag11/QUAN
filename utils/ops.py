# utils/quaternion_ops.py

import torch
from itertools import product
import seaborn
import numpy as np 
import shapely.geometry
import math
import torch.nn.functional as F


def bbox2dist(anchor_points, bbox_targets, reg_max=16, xywh=True):
    """Convert bounding box coordinates to distance targets using DFL style encoding."""
    is_batched = bbox_targets.dim() == 3
    if not is_batched:
        bbox_targets = bbox_targets.unsqueeze(0)
        anchor_points = anchor_points.unsqueeze(0)
    
    # Get target coordinates
    x_t, y_t, w_t, h_t = bbox_targets.unbind(-1)  # [B, M]
    x_a, y_a = anchor_points.unbind(-1)  # [B, N]
    
    # Prepare for broadcasting
    B, M = x_t.shape
    N = x_a.shape[1]
    x_t = x_t.unsqueeze(-1)  # [B, M, 1]
    y_t = y_t.unsqueeze(-1)  # [B, M, 1]
    x_a = x_a.unsqueeze(1)   # [B, 1, N]
    y_a = y_a.unsqueeze(1)   # [B, 1, N]
    
    # Compute relative target boxes
    xy_diff = torch.stack([
        x_t - x_a,  # [B, M, N]
        y_t - y_a   # [B, M, N]
    ], dim=-1)  # [B, M, N, 2]
    
    # Find closest anchor for each target
    dist_xy = (xy_diff ** 2).sum(-1)  # [B, M, N]
    closest_anchor = dist_xy.argmin(dim=-1)  # [B, M]
    
    # Initialize output distributions
    dist = torch.zeros(B, N, 4 * reg_max, device=bbox_targets.device)
    
    # Create reference points for DFL
    ref_points = torch.linspace(0, 1, reg_max, device=bbox_targets.device)
    
    for b in range(B):
        for m in range(M):
            a = closest_anchor[b, m]
            
            # Get deltas for this target-anchor pair
            dx = xy_diff[b, m, a, 0].item()
            dy = xy_diff[b, m, a, 1].item()
            dw = w_t[b, m].log().item()
            dh = h_t[b, m].log().item()
            
            # Scale values to [0, 1] range
            dx = (dx / 8.0 + 1) / 2  # Map [-8, 8] to [0, 1]
            dy = (dy / 8.0 + 1) / 2
            dw = (dw + 4) / 8       # Map [-4, 4] to [0, 1]
            dh = (dh + 4) / 8
            
            # Clamp values to valid range
            dx = max(0, min(0.999, dx))
            dy = max(0, min(0.999, dy))
            dw = max(0, min(0.999, dw))
            dh = max(0, min(0.999, dh))
            
            # Find left and right reference points
            dx_idx = int(dx * (reg_max - 1))
            dy_idx = int(dy * (reg_max - 1))
            dw_idx = int(dw * (reg_max - 1))
            dh_idx = int(dh * (reg_max - 1))
            
            # Compute weights for left and right points
            dx_w = dx * (reg_max - 1) - dx_idx
            dy_w = dy * (reg_max - 1) - dy_idx
            dw_w = dw * (reg_max - 1) - dw_idx
            dh_w = dh * (reg_max - 1) - dh_idx
            
            # Set distribution values
            if dx_idx < reg_max - 1:
                dist[b, a, dx_idx] = 1 - dx_w
                dist[b, a, dx_idx + 1] = dx_w
            else:
                dist[b, a, dx_idx] = 1
                
            if dy_idx < reg_max - 1:
                dist[b, a, reg_max + dy_idx] = 1 - dy_w
                dist[b, a, reg_max + dy_idx + 1] = dy_w
            else:
                dist[b, a, reg_max + dy_idx] = 1
                
            if dw_idx < reg_max - 1:
                dist[b, a, 2 * reg_max + dw_idx] = 1 - dw_w
                dist[b, a, 2 * reg_max + dw_idx + 1] = dw_w
            else:
                dist[b, a, 2 * reg_max + dw_idx] = 1
                
            if dh_idx < reg_max - 1:
                dist[b, a, 3 * reg_max + dh_idx] = 1 - dh_w
                dist[b, a, 3 * reg_max + dh_idx + 1] = dh_w
            else:
                dist[b, a, 3 * reg_max + dh_idx] = 1
    
    return dist.squeeze(0) if not is_batched else dist

def dist2bbox(distances, anchor_points, stride=None, xywh=True, apply_softmax=False):
    """Convert distance predictions back to boxes using DFL style decoding."""
    is_batched = distances.dim() == 3
    if not is_batched:
        distances = distances.unsqueeze(0)
        anchor_points = anchor_points.unsqueeze(0)
    
    reg_max = distances.shape[-1] // 4
    
    # Clear any cached tensors
    torch.cuda.empty_cache()
    
    # Get batch size and ensure shapes match within this batch
    B, N = distances.shape[:2]
    anchor_points = anchor_points[:B]  # Ensure we only use anchors for current batch
    
    # Get anchor points - handle possible shape mismatches
    x_a, y_a = anchor_points.unbind(-1)  # [B, N_anchors]
    
    # Match shapes if needed
    if x_a.shape[1] != N:
        # Interpolate anchor points to match distances if sizes differ
        x_a = F.interpolate(x_a.unsqueeze(1), size=N, mode='nearest').squeeze(1)
        y_a = F.interpolate(y_a.unsqueeze(1), size=N, mode='nearest').squeeze(1)
    
    # Reshape distances
    distances = distances.reshape(B, N, 4, reg_max).contiguous()
    
    if apply_softmax:
        distances = F.softmax(distances, dim=-1)
    
    # Create reference points
    ref = torch.linspace(0, 1, reg_max, device=distances.device)
    
    # Compute deltas
    dx = (distances[..., 0, :] * ref).sum(dim=-1)  # [B, N]
    dy = (distances[..., 1, :] * ref).sum(dim=-1)
    dw = (distances[..., 2, :] * ref).sum(dim=-1)
    dh = (distances[..., 3, :] * ref).sum(dim=-1)
    
    # Scale back to original range
    dx = (2 * dx - 1) * 8  # Map [0, 1] to [-8, 8]
    dy = (2 * dy - 1) * 8
    dw = 8 * dw - 4        # Map [0, 1] to [-4, 4]
    dh = 8 * dh - 4
    
    # Compute final coordinates
    x = x_a + dx
    y = y_a + dy
    w = torch.exp(dw)
    h = torch.exp(dh)
    
    # Apply stride if provided
    if stride is not None:
        x = x * stride
        y = y * stride
        w = w * stride
        h = h * stride
    
    # Stack output
    if xywh:
        boxes = torch.stack([x, y, w, h], -1)
    else:
        boxes = torch.stack([
            x - w/2, y - h/2,
            x + w/2, y + h/2
        ], -1)
    
    return boxes.squeeze(0) if not is_batched else boxes



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

