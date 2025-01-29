# loss/box_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh, dist2bbox, dist2rbox, make_anchors
from utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner
from utils.torch_utils import autocast

from utils.metrics import bbox_iou, stable_bbox_iou, probiou
from utils.ops import bbox2dist
import math 
from typing import List, Tuple

__all__ = ["QuaternionLoss"]

def enforce_quaternion_hemisphere(quat):
    """Ensure the scalar part of the quaternion is non-negative."""
    mask = quat[..., 3:] < 0
    quat = quat.clone()
    quat[mask] = -quat[mask]
    return quat



class ClassificationLoss(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.nc = nc
        self.cls_criterion = nn.CrossEntropyLoss(reduction='none')
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc)

    def forward(self, pred_cls, pred_reg, targets, anchor_points):
        device = pred_cls[0].device
        total_loss = torch.zeros(1, device=device)
        total_correct = 0
        total_samples = 0
        stride = [8, 16, 32]
        anchor_indices = [0]  # Start index for each level
        for cls_out in pred_cls:
            anchor_indices.append(anchor_indices[-1] + cls_out.shape[1])

        for level_idx, (level_cls, level_reg) in enumerate(zip(pred_cls, pred_reg)):
            B, HW, _ = level_cls.shape
            
            # Get anchor points for this level
            start_idx = anchor_indices[level_idx]
            end_idx = anchor_indices[level_idx + 1]
            level_anchors = anchor_points[start_idx:end_idx]  # Get anchors for just this level
            
            
            for b in range(B):
                gt_boxes = targets['boxes'][b]
                gt_labels = targets['labels'][b]
                
                if len(gt_boxes) == 0:
                    continue

                # Get assignments from TaskAlignedAssigner
                target_labels, target_scores, fg_mask = self.assigner(
                    torch.softmax(level_cls[b], dim=1),  # Convert logits to probabilities
                    level_reg[b],
                    level_anchors,
                    gt_labels,
                    gt_boxes,
                    stride[level_idx]
                )

                # Compute loss only for positive samples
                if fg_mask.sum() > 0:
                    loss = self.cls_criterion(
                        level_cls[b][fg_mask],
                        target_labels[fg_mask]
                    )
                    total_loss += loss.sum()
                    total_samples += fg_mask.sum().item()

        return {
            'loss': total_loss / max(total_samples, 1),
            'num_pos': total_samples
        }
    def _debug_shapes(self, msg, **tensors):
        """Helper to debug tensor shapes."""
        print(f"\n{msg}:")
        for name, tensor in tensors.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{name}: {tensor.shape}")
            elif isinstance(tensor, (list, tuple)):
                print(f"{name}: {[t.shape if isinstance(t, torch.Tensor) else None for t in tensor]}")

def debug_predictions(pred_cls, pred_reg, targets=None, conf_threshold=0.25):
    """
    Detailed debugging of predictions during inference.
    
    Args:
        pred_cls (list): List of classification predictions [B, H*W, nc] for each level
        pred_reg (list): List of regression predictions [B, H*W, 4] for each level
        targets (dict, optional): Ground truth targets for validation
        conf_threshold (float): Confidence threshold for predictions
    """
    for level_idx, (level_cls, level_reg) in enumerate(zip(pred_cls, pred_reg)):
        print(f"\nFeature Level {level_idx}:")
        B, HW, nc = level_cls.shape
        
        print(f"Shape Analysis:")
        print(f"- Classification output: {level_cls.shape}")
        print(f"- Regression output: {level_reg.shape}")
        
        for b in range(B):
            print(f"\nBatch {b}:")
            
            # Apply sigmoid to get probabilities
            scores = torch.sigmoid(level_cls[b])  # [HW, nc]
            boxes = level_reg[b]  # [HW, 4]
            
            # Get max scores and corresponding classes
            max_scores, pred_classes = scores.max(dim=1)  # [HW]
            
            # Filter by confidence
            confident_mask = max_scores > conf_threshold
            num_confident = confident_mask.sum().item()
            
            print(f"Statistics:")
            print(f"- Total predictions: {HW}")
            print(f"- Confident predictions (>{conf_threshold}): {num_confident}")
            
            if num_confident > 0:
                # Get confident predictions
                conf_scores = max_scores[confident_mask]
                conf_classes = pred_classes[confident_mask]
                conf_boxes = boxes[confident_mask]
                
                # Print top 5 predictions
                top_k = min(5, num_confident)
                top_indices = conf_scores.argsort(descending=True)[:top_k]
                
                print("\nTop 5 Predictions:")
                for idx in top_indices:
                    print(f"Class {conf_classes[idx].item()}: "
                          f"Score {conf_scores[idx]:.4f}, "
                          f"Box {conf_boxes[idx].tolist()}")
            
            # Compare with ground truth if available
            if targets is not None:
                gt_boxes = targets['boxes'][b]
                gt_labels = targets['labels'][b]
                
                print("\nGround Truth:")
                print(f"- Number of GT objects: {len(gt_boxes)}")
                if len(gt_boxes) > 0:
                    for gt_idx in range(min(5, len(gt_boxes))):
                        print(f"GT {gt_idx}: "
                              f"Class {gt_labels[gt_idx].item()}, "
                              f"Box {gt_boxes[gt_idx].tolist()}")

            # Box statistics
            if boxes.numel() > 0:
                print("\nBox Statistics:")
                print(f"- Mean box size: {boxes[:, 2:].mean():.4f}")
                print(f"- Box coordinate ranges:")
                print(f"  x: [{boxes[:, 0].min():.2f}, {boxes[:, 0].max():.2f}]")
                print(f"  y: [{boxes[:, 1].min():.2f}, {boxes[:, 1].max():.2f}]")
                print(f"  w: [{boxes[:, 2].min():.2f}, {boxes[:, 2].max():.2f}]")
                print(f"  h: [{boxes[:, 3].min():.2f}, {boxes[:, 3].max():.2f}]")
            
            print("-" * 50)


def get_classification_metrics(pred_cls, targets, nc):
    """
    Compute detailed classification metrics with error handling
    
    Args:
        pred_cls (list): Classification predictions from each level
        targets (dict): Ground truth targets
        nc (int): Number of classes
    
    Returns:
        dict: Detailed metrics
    """
    device = pred_cls[0].device
    total_correct = 0
    total_samples = 0
    per_class_correct = {}
    per_class_total = {}
    
    for level_preds in pred_cls:
        B, HW, nc_pred = level_preds.shape
        assert nc_pred == nc, f"Prediction channels {nc_pred} do not match expected {nc}"
        
        for b in range(B):
            gt_labels = targets['labels'][b]
            
            if len(gt_labels) == 0:
                continue
            
            # Apply softmax to get probabilities
            probs = F.softmax(level_preds[b], dim=-1)
            avg_pred = probs.mean(dim=0)  # Average across spatial locations
            
            # Sanity check labels
            valid_labels = [
                label.item() for label in gt_labels 
                if 0 <= label.item() < nc
            ]
            
            if not valid_labels:
                print(f"No valid labels for batch {b}")
                continue
            
            pred_class = avg_pred.argmax().item()
            
            # Track per-class metrics
            for label in valid_labels:
                per_class_total[label] = per_class_total.get(label, 0) + 1
                
                if pred_class == label:
                    total_correct += 1
                    per_class_correct[label] = per_class_correct.get(label, 0) + 1
                
                total_samples += 1
    
    # Compute per-class metrics
    per_class_accuracy = {}
    for label, total in per_class_total.items():
        correct = per_class_correct.get(label, 0)
        per_class_accuracy[label] = correct / total * 100
    
    return {
        'overall_accuracy': total_correct / max(total_samples, 1) * 100,
        'correct': total_correct,
        'total_samples': total_samples,
        'per_class_accuracy': per_class_accuracy
    }

class DetectionLoss(nn.Module):
    def __init__(self, nc=80):
        super().__init__()
        self.nc = nc
        self.cls_criterion = nn.CrossEntropyLoss(reduction='none')
        self.box_criterion = nn.L1Loss(reduction='none')
    
    def forward(self, pred_cls, pred_reg, targets):
        """
        Args:
            pred_cls (list[Tensor]): Classification logits [B, HW, nc] for each level
            pred_reg (list[Tensor]): Box coordinates [B, HW, 4] for each level
            targets (dict): Contains:
                boxes (Tensor): [B, num_objects, 4]
                labels (Tensor): [B, num_objects] - integer class labels
        """
        device = pred_cls[0].device
        num_levels = len(pred_cls)
        
        # Initialize losses
        cls_loss = torch.zeros(1, device=device)
        reg_loss = torch.zeros(1, device=device)
        num_pos = 0  # Counter for positive samples
        
        for level in range(num_levels):
            # Get predictions for this level
            lvl_cls = pred_cls[level]  # [B, HW, nc]
            lvl_reg = pred_reg[level]  # [B, HW, 4]
            
            B, HW, _ = lvl_cls.shape
            
            # Process each batch
            for b in range(B):
                gt_boxes = targets['boxes'][b]    # [num_obj, 4]
                gt_labels = targets['labels'][b]   # [num_obj]
                
                if len(gt_boxes) == 0:
                    continue
                
                # Get predicted box centers
                pred_xy = lvl_reg[b, :, :2]  # [HW, 2]
                
                # Compute center-based distances
                gt_xy = gt_boxes[:, :2]  # [num_obj, 2]
                dist_matrix = torch.cdist(gt_xy, pred_xy)  # [num_obj, HW]
                
                # Assign targets to predictions
                min_dists, matches = dist_matrix.min(dim=1)  # [num_obj]
                
                # Only keep matches within threshold
                valid_mask = min_dists < 2.5  # Typical assign radius
                valid_matches = matches[valid_mask]
                
                if len(valid_matches) == 0:
                    continue
                
                # Classification loss - only for matched positions
                cls_loss += self.cls_criterion(
                    lvl_cls[b, valid_matches],  # [num_valid, nc]
                    gt_labels[valid_mask]       # [num_valid]
                ).sum()
                
                # Box regression loss - only for matched positions
                reg_loss += self.box_criterion(
                    lvl_reg[b, valid_matches],  # [num_valid, 4]
                    gt_boxes[valid_mask]        # [num_valid, 4]
                ).sum()
                
                num_pos += len(valid_matches)
        
        # Normalize losses by number of positive samples
        num_pos = max(num_pos, 1)  # Avoid division by zero
        
        return dict(
            loss_cls=cls_loss / num_pos,
            loss_box=reg_loss / num_pos,
            num_pos=num_pos
        )

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

def get_predictions(pred_cls, pred_box, conf_thresh=0.25):
    """Convert predictions to detections.
    
    Args:
        pred_cls: List of classification predictions [B, nc, H, W]
        pred_box: List of box predictions [B, 4, H, W]
        conf_thresh: Confidence threshold
    
    Returns:
        tuple: (boxes, classes, scores)
    """
    all_boxes = []
    all_scores = []
    all_classes = []

    # Process each feature level
    for cls_pred, box_pred in zip(pred_cls, pred_box):
        # Get probability scores
        scores = F.softmax(cls_pred, dim=1)  # [B, nc, H, W]
        
        # Get max confidence and class for each position
        max_scores, max_classes = scores.max(dim=1)  # [B, H, W]
        
        # Get boxes above threshold
        mask = max_scores > conf_thresh
        if mask.any():
            # Reshape predictions to [N, dims]
            boxes = box_pred.permute(0, 2, 3, 1)[mask]  # [N, 4]
            scores = max_scores[mask]  # [N]
            classes = max_classes[mask]  # [N]
            
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(classes)

    # Concatenate all levels
    if len(all_boxes) > 0:
        boxes = torch.cat(all_boxes)
        scores = torch.cat(all_scores)
        classes = torch.cat(all_classes)
        return boxes, classes, scores
    else:
        return (torch.zeros(0, 4), torch.zeros(0), torch.zeros(0))
    
# class DetectionLoss(nn.Module):
#     """
#     Detection loss for quaternion-based model without anchors.
#     Ground truth boxes and labels are in original format (not normalized).
#     """
#     def __init__(self, nc=80):
#         super().__init__()
#         self.nc = nc
        
#     def forward(self, outputs, targets):
#         device = outputs[0][0].device
#         loss_box = torch.zeros(1, device=device)
#         loss_cls = torch.zeros(1, device=device)        
        
#         for cls_output, box_output in zip(outputs[0], outputs[1]):
#             B, C, Q, H, W = cls_output.shape
            
#             # Reshape predictions
#             pred_cls = cls_output.permute(0, 2, 3, 4, 1).reshape(B, -1, self.nc)  # [B, H*W*4, nc]
#             pred_box = box_output.permute(0, 2, 3, 4, 1).reshape(B, -1, 4)  # [B, H*W*4, 4]
            
#             for b in range(B):
#                 gt_boxes = targets['boxes'][b]     # Original box coordinates
#                 gt_labels = targets['labels'][b]   # Integer class labels
                
#                 if len(gt_boxes) == 0:
#                     continue


#                 loss_cls += F.cross_entropy(pred_cls[b], gt_labels)

#                 # Compute IoU between predictions and ground truth
#                 ious = stable_bbox_iou(pred_box[b], gt_boxes, xywh=True)
#                 best_ious, best_target_idx = ious.max(dim=1)
#                 pos_mask = best_ious > 0.5
                
#                 if pos_mask.sum() > 0:
#                     # Box loss - directly compute L1 loss on coordinates
#                     pos_pred_boxes = pred_box[b][pos_mask]
#                     pos_target_boxes = gt_boxes[best_target_idx[pos_mask]]
#                     loss_box += F.l1_loss(pos_pred_boxes, pos_target_boxes)
                    
#                     # # Classification loss - pass raw logits to cross_entropy
#                     # loss_cls += F.cross_entropy(
#                     #     pred_cls[b][pos_mask], 
#                     #     gt_labels[best_target_idx[pos_mask]]
#                     # )

#         num_levels = len(outputs[0])
#         loss_box = loss_box / num_levels
#         loss_cls = loss_cls / num_levels
        
#         return loss_box, None, loss_cls, None