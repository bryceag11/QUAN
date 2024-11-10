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

class TaskAlignedAssigner:
    """
    General Task-Aligned Assigner for OBBs with quaternions.
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
