# utils/quaternion_ops.py

import torch
from torch import nn
from .metrics import bbox_iou


class RotatedTaskAlignedAssigner:
    """
    Rotated Task-Aligned Assigner for OBBs with quaternions.
    """
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0):
        """
        Initialize the assigner.

        Args:
            topk (int): Top-k selection for assignment.
            num_classes (int): Number of classes.
            alpha (float): Weight for classification score.
            beta (float): Weight for IoU.
        """
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

    def assign(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """
        Assign ground truth to predictions.

        Args:
            pred_scores (torch.Tensor): Predicted class scores, shape (N, C).
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (N, 4).
            anchor_points (torch.Tensor): Anchor points, shape (N, 2).
            gt_labels (torch.Tensor): Ground truth class labels, shape (M, 1).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (M, 4).
            mask_gt (torch.Tensor): Mask indicating valid ground truths, shape (M, 1).

        Returns:
            Tuple: Assigned bounding boxes, scores, and masks.
        """
        # Compute IoU between predicted bboxes and ground truth
        iou_matrix = bbox_iou(pred_bboxes, gt_bboxes, quats1=None, quats2=None, xywh=False)
        
        # Assign topk anchors for each gt
        topk_iou, topk_indices = torch.topk(iou_matrix, self.topk, dim=0)
        
        # Compute task aligned score
        scores = (pred_scores[:, gt_labels.squeeze(1)] ** self.alpha) * (topk_iou ** self.beta)
        
        # Get the best anchor for each gt
        best_scores, best_indices = torch.max(scores, dim=0)
        
        # Assign
        fg_mask = torch.zeros(pred_scores.shape[0], dtype=torch.bool, device=pred_scores.device)
        fg_mask[best_indices] = True
        
        assigned_bboxes = gt_bboxes
        assigned_scores = best_scores
        
        return None, assigned_bboxes, assigned_scores, fg_mask, None

# class TaskAlignedAssigner:
#     """
#     General Task-Aligned Assigner for OBBs with quaternions.
#     """
#     def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0):
#         """
#         Initialize the assigner.

#         Args:
#             topk (int): Top-k selection for assignment.
#             num_classes (int): Number of classes.
#             alpha (float): Weight for classification score.
#             beta (float): Weight for IoU.
#         """
#         self.topk = topk
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta

#     def assign(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
#         """
#         Assign ground truth to predictions.

#         Args:
#             pred_scores (torch.Tensor): Predicted class scores, shape (N, C).
#             pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (N, 4).
#             anchor_points (torch.Tensor): Anchor points, shape (N, 2).
#             gt_labels (torch.Tensor): Ground truth class labels, shape (M, 1).
#             gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (M, 4).
#             mask_gt (torch.Tensor): Mask indicating valid ground truths, shape (M, 1).

#         Returns:
#             Tuple: Assigned bounding boxes, scores, and masks.
#         """
#         # Compute IoU between predicted bboxes and ground truth
#         iou_matrix = bbox_iou(pred_bboxes, gt_bboxes, quats1=None, quats2=None, xywh=False)
        
#         # Assign topk anchors for each gt
#         topk_iou, topk_indices = torch.topk(iou_matrix, self.topk, dim=0)
        
#         # Compute task aligned score
#         scores = (pred_scores[:, gt_labels.squeeze(1)] ** self.alpha) * (topk_iou ** self.beta)
        
#         # Get the best anchor for each gt
#         best_scores, best_indices = torch.max(scores, dim=0)
        
#         # Assign
#         fg_mask = torch.zeros(pred_scores.shape[0], dtype=torch.bool, device=pred_scores.device)
#         fg_mask[best_indices] = True
        
#         assigned_bboxes = gt_bboxes
#         assigned_scores = best_scores
        
#         return None, assigned_bboxes, assigned_scores, fg_mask, None


class TaskAlignedAssigner:
    def __init__(self, topk=10, num_classes=80):
       self.topk = topk
       self.num_classes = num_classes
       
    def decode_boxes(self, reg_preds, anchors, stride):
        """
        Convert regression predictions to boxes with GT-matched scaling
        """
        # Get anchor points
        x_center = anchors[:, 0]
        y_center = anchors[:, 1]
        
        # Much larger scale factors based on GT analysis
        CENTER_SCALE = 8.0    # For center offset
        MIN_SIZE = stride * 4 # Minimum box size
        SIZE_MULTIPLIER = 16.0  # Multiplier for width/height
        
        # Decode centers with larger offsets
        x_center_pred = x_center + reg_preds[:, 0] * stride * CENTER_SCALE
        y_center_pred = y_center + reg_preds[:, 1] * stride * CENTER_SCALE
        
        # Decode width and height with base size + exponential scaling
        w_pred = MIN_SIZE + SIZE_MULTIPLIER * stride * torch.exp(reg_preds[:, 2])
        h_pred = MIN_SIZE + SIZE_MULTIPLIER * stride * torch.exp(reg_preds[:, 3])
        
        # Convert to xyxy format
        boxes = torch.stack([
            x_center_pred - w_pred/2,  # x1
            y_center_pred - h_pred/2,  # y1
            x_center_pred + w_pred/2,  # x2
            y_center_pred + h_pred/2   # y2
        ], dim=-1)
        
        # Ensure boxes stay within image bounds
        boxes = boxes.clamp(min=0, max=640)  # Assuming 640x640 image
        
        print("\nDecoding Statistics:")
        print(f"Center predictions - X: {x_center_pred.min():.1f} to {x_center_pred.max():.1f}")
        print(f"Center predictions - Y: {y_center_pred.min():.1f} to {y_center_pred.max():.1f}")
        print(f"Width predictions: {w_pred.min():.1f} to {w_pred.max():.1f}")
        print(f"Height predictions: {h_pred.min():.1f} to {h_pred.max():.1f}")
        
        print("\nBox Size Analysis:")
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        print(f"Final box widths: {box_widths.min():.1f} to {box_widths.max():.1f}")
        print(f"Final box heights: {box_heights.min():.1f} to {box_heights.max():.1f}")
        
        return boxes
        
    def __call__(self, pred_scores, pred_boxes, anchors, gt_labels, gt_boxes, stride):
        """
        Modified to ensure GT boxes are properly formatted, with debugging prints
        """
        # Decode predicted boxes
        decoded_boxes = self.decode_boxes(pred_boxes, anchors, stride)
        
        print("\nBox Coordinate Analysis:")
        print("Original GT boxes first 3:")
        for i in range(min(3, len(gt_boxes))):
            print(f"GT {i}: {gt_boxes[i].tolist()}")
        
        # Fix GT boxes - ensure proper coordinate ordering
        fixed_gt_boxes = torch.stack([
            torch.min(gt_boxes[..., 0], gt_boxes[..., 2]),  # x1
            torch.min(gt_boxes[..., 1], gt_boxes[..., 3]),  # y1
            torch.max(gt_boxes[..., 0], gt_boxes[..., 2]),  # x2
            torch.max(gt_boxes[..., 1], gt_boxes[..., 3])   # y2
        ], dim=-1)
        
        print("\nFixed GT boxes first 3:")
        for i in range(min(3, len(fixed_gt_boxes))):
            print(f"Fixed GT {i}: {fixed_gt_boxes[i].tolist()}")
        
        print("\nDecoded boxes first 3:")
        for i in range(min(3, len(decoded_boxes))):
            print(f"Decoded {i}: {decoded_boxes[i].tolist()}")
        
        # Compute IoUs with proper unsqueeze operations
        decoded_boxes_unsqueezed = decoded_boxes.unsqueeze(1)  # [N, 1, 4]
        fixed_gt_boxes_unsqueezed = fixed_gt_boxes.unsqueeze(0)  # [1, M, 4]
        
        print(f"\nShape before IoU computation:")
        print(f"Decoded boxes shape: {decoded_boxes_unsqueezed.shape}")
        print(f"GT boxes shape: {fixed_gt_boxes_unsqueezed.shape}")
        
        # Compute IoUs
        ious = bbox_iou(decoded_boxes_unsqueezed, fixed_gt_boxes_unsqueezed, xywh=False)
        
        print("\nIoU Analysis:")
        print(f"IoU matrix shape: {ious.shape}")
        print(f"IoU range: {ious.min():.4f} to {ious.max():.4f}")
        
        # Sample some IoU values
        print("\nSample IoU values (first 3 predictions vs first 3 GT boxes):")
        for i in range(min(3, ious.shape[0])):
            for j in range(min(3, ious.shape[1])):
                print(f"IoU between pred {i} and GT {j}: {ious[i,j]:.4f}")

        # Get best predictions for each GT based on IoU
        num_gt = len(gt_labels)
        K = min(self.topk, ious.size(0))  # Can't assign more preds than we have
        values, indices = ious.topk(K, dim=0)  # Take top K matches for each GT

        # Create assignment mask
        pos_mask = torch.zeros_like(pred_scores[:, 0], dtype=torch.bool)

        # Initialize target arrays
        target_labels = torch.full_like(pred_scores[:, 0], -1, dtype=torch.long)  # Same type as gt_labels
        target_scores = torch.zeros_like(pred_scores[:, 0])  # For storing confidence scores

        # Assign each GT to top K predictions
        for gt_idx in range(num_gt):
            # Get indices of top K predictions for this GT
            matching_pred_idxs = indices[:, gt_idx]
            
            # Mark these predictions as positive
            pos_mask[matching_pred_idxs] = True
            
            # Assign GT label to these predictions
            target_labels[matching_pred_idxs] = gt_labels[gt_idx]
            
            # Store prediction confidence as target score
            match_scores = pred_scores[matching_pred_idxs, gt_labels[gt_idx]]
            target_scores[matching_pred_idxs] = match_scores

        return target_labels, target_scores, pos_mask

#    def delta2box(self, deltas, anchors, stride):
#        """
#        Convert box deltas and anchors to pred boxes.
       
#        Args:
#            deltas: [num_queries, 4] - predicted box deltas
#            anchors: [num_queries, 2] - anchor points
           
#        Returns:
#            pred_boxes: [num_queries, 4] in [x1,y1,x2,y2] format
#        """
#        # Scale anchors first (they are in feature-level coordinates)
#        # Note: actual stride value needed here
#        anchors = anchors * stride
       
#        # deltas are in form [dx,dy,dw,dh]
#        # Convert to center form
#        pred_ctr = anchors + deltas[:, :2] * stride  # Add scaled deltas to anchor centers
       
#        # Width and height
#        pred_wh = torch.exp(deltas[:, 2:]) * stride  # Scale width/height by stride
       
#        # Convert to [x1,y1,x2,y2] format
#        pred_boxes = torch.cat([
#            pred_ctr - 0.5 * pred_wh,  # x1,y1 
#            pred_ctr + 0.5 * pred_wh   # x2,y2
#        ], dim=1)
       
#        return pred_boxes

