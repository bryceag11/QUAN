# loss/box_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ops import crop_mask, xywh2xyxy, xyxy2xywh, dist2bbox, dist2rbox, make_anchors
from utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner
from utils.torch_utils import autocast

from utils.metrics import bbox_iou, probiou
from utils.ops import bbox2dist

__all__ = ["QuaternionLoss"]

def enforce_quaternion_hemisphere(quat):
    """Ensure the scalar part of the quaternion is non-negative."""
    mask = quat[..., 3:] < 0
    quat = quat.clone()
    quat[mask] = -quat[mask]
    return quat

class QuaternionLoss(nn.Module):
    """Custom loss function for quaternion-based object detection."""

    def __init__(self, obj_scale=1.0, cls_scale=1.0, box_scale=1.0, angle_scale=1.0):
        super(QuaternionLoss, self).__init__()
        self.obj_scale = obj_scale
        self.cls_scale = cls_scale
        self.box_scale = box_scale
        self.angle_scale = angle_scale
        self.bce = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
        self.angle_loss = nn.MSELoss()  # Or another suitable loss for angles/quaternions

    def forward(self, predictions, targets):
        # Split predictions and targets into respective components
        pred_boxes, pred_classes, pred_angles = predictions
        target_boxes, target_classes, target_angles = targets

        # Objectness loss
        obj_loss = self.bce(pred_classes, target_classes)

        # Bounding box loss
        box_loss = self.l1(pred_boxes, target_boxes)

        # Orientation loss
        angle_loss = self.angle_loss(pred_angles, target_angles)

        # Total loss
        total_loss = (self.obj_scale * obj_loss +
                      self.cls_scale * box_loss +
                      self.angle_scale * angle_loss)
        
        return total_loss

# Primarily deals with classification by focusing on positive samples and modulating loss based on prediction confidence
class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss

class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e., FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")

        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor

        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor

        return loss.mean(1).sum()
class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL) for bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        """
        Compute the Distribution Focal Loss.

        Args:
            pred_dist (torch.Tensor): Predicted distance distributions, shape (N, reg_max).
            target (torch.Tensor): Target distances, shape (N,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        target = target.clamp(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl

        loss = (
            F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr, reduction='none') * wr
        )
        return loss.mean()


    @staticmethod
    def quaternion_angular_loss(pred_quat, target_quat):
        """
        Compute angular loss between predicted and target quaternions.
        Enforce hemisphere consistency before calculation.

        Args:
            pred_quat (torch.Tensor): Predicted quaternions, shape (..., 4)
            target_quat (torch.Tensor): Target quaternions, shape (..., 4)

        Returns:
            torch.Tensor: Angular loss
        """
        # Enforce hemisphere consistency
        pred_quat = enforce_quaternion_hemisphere(pred_quat)
        target_quat = enforce_quaternion_hemisphere(target_quat)

        # Ensure quaternions are normalized
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        target_quat = F.normalize(target_quat, p=2, dim=-1)

        # Compute the dot product
        dot_product = (pred_quat * target_quat).sum(dim=-1)
        dot_product = torch.clamp(dot_product, min=-1.0, max=1.0)
        # Compute angular loss
        angular_loss = torch.acos(torch.abs(dot_product))  # radians
        return angular_loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, nc=80, use_quat=False):
        super().__init__()
        self.reg_max = reg_max
        self.nc = nc
        self.use_dfl = reg_max > 1
        self.use_quat = use_quat
        self.dfl_loss = DFLoss(reg_max) if self.use_dfl else None
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
        if use_quat:
            self.quat_reg_loss = QuaternionRegularizationLoss(lambda_reg=0.1)
            self.geo_consistency_loss = GeometricConsistencyLoss(lambda_geo=0.5)
            self.orientation_smoothness_loss = OrientationSmoothnessLoss(lambda_smooth=0.3)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, 
                target_scores_sum, fg_mask, pred_quat=None, target_quat=None):
        """
        Calculate all components of the box loss.
        
        Args:
            pred_dist: shape (batch_size, num_anchors, 4 * reg_max)
            pred_bboxes: shape (batch_size, num_anchors, 4)
            anchor_points: shape (num_anchors, 2)
            target_bboxes: shape (batch_size, num_anchors, 4)
            target_scores: shape (batch_size, num_anchors)
            target_scores_sum: scalar
            fg_mask: shape (batch_size, num_anchors)
        """
        # Initialize losses
        device = pred_dist.device
        box_loss = torch.tensor(0., device=device)
        dfl_loss = torch.tensor(0., device=device)
        quat_loss = torch.tensor(0., device=device)

        if fg_mask.sum():
            # Expand anchor points to match batch dimension
            batch_size = pred_dist.shape[0]
            expanded_anchor_points = anchor_points.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Calculate DFL loss for each component
            if self.use_dfl:
                dist_pred_pos = pred_dist[fg_mask]  # Shape: (num_pos, 4 * reg_max)
                anchor_points_pos = expanded_anchor_points[fg_mask]  # Shape: (num_pos, 2)
                target_bboxes_pos = target_bboxes[fg_mask]  # Shape: (num_pos, 4)
                
                # Get target distances
                dist_targets = bbox2dist(anchor_points_pos, target_bboxes_pos, self.reg_max-1)  # Shape: (num_pos, 4)
                
                # Apply DFLLoss to each distance component
                dfl_loss = 0.0
                for c in range(4):
                    dfl_loss += self.dfl_loss(
                        dist_pred_pos[:, c * self.reg_max:(c + 1) * self.reg_max],
                        dist_targets[:, c]
                    )
                dfl_loss = dfl_loss / 4.0  # Average over the four components
            
            # Calculate IoU loss
            matched_pred_boxes = pred_bboxes[fg_mask]
            matched_target_boxes = target_bboxes[fg_mask]
            iou = bbox_iou(matched_pred_boxes, matched_target_boxes, xywh=True)
            box_loss = ((1.0 - iou) * target_scores[fg_mask]).sum() / target_scores_sum
            
            # Calculate quaternion losses if enabled
            if self.use_quat and pred_quat is not None and target_quat is not None:
                matched_pred_quat = pred_quat[fg_mask]
                matched_target_quat = target_quat[fg_mask]
                
                quat_reg = self.quat_reg_loss(matched_pred_quat)
                geo_consistency = self.geo_consistency_loss(matched_pred_boxes, matched_pred_quat)
                orientation_smoothness = self.orientation_smoothness_loss(
                    matched_pred_quat,
                    self.get_neighbor_quat(matched_pred_quat)
                )
                quat_loss = quat_reg + geo_consistency + orientation_smoothness

        return box_loss, dfl_loss, quat_loss

    def get_neighbor_quat(self, pred_quat):
        """Get neighboring quaternions for smoothness calculation."""
        # Simple implementation - can be enhanced based on your needs
        return torch.roll(pred_quat, shifts=1, dims=0)

    @staticmethod
    def make_anchors(h, w, device):
        """Generate anchor points with proper offset and clamping."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        grid_xy = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1)
        ], dim=1)
        
        # Add offset to center the anchors
        grid_xy += 0.5
        
        # Clamp anchor points to prevent them from going out of image boundaries
        grid_xy = torch.clamp(grid_xy, min=0.0, max=max(h, w))
        
        return grid_xy



    def bbox_decode(self, pred_dist, anchor_points):
        """
        Decode predicted distributions to bounding boxes with numerical stability.
        
        Args:
            pred_dist: shape (batch_size, num_anchors, 4)
            anchor_points: shape (num_anchors, 2)
        
        Returns:
            decoded_bboxes: shape (batch_size, num_anchors, 4)
        """
        batch_size = pred_dist.size(0) if pred_dist.dim() == 3 else 1
        num_anchors = pred_dist.size(-2)
        
        if self.use_dfl and pred_dist is not None:
            # Reshape for softmax: (batch_size, num_anchors, 4, reg_max)
            pred_dist = pred_dist.reshape(batch_size, num_anchors, 4, self.reg_max)
            
            # Clamp pred_dist to prevent extreme values before softmax
            pred_dist = torch.clamp(pred_dist, min=-10.0, max=10.0)
            
            # Apply softmax along the reg_max dimension
            pred_dist = pred_dist.softmax(dim=3)
            
            # Create distance array
            distance_array = torch.arange(self.reg_max, device=pred_dist.device, dtype=pred_dist.dtype)
            
            # Calculate distances: (batch_size, num_anchors, 4)
            pred_dist = (pred_dist * distance_array).sum(dim=3)
            
            # Clamp pred_dist to prevent extreme values after sum
            pred_dist = torch.clamp(pred_dist, min=0.0, max=1e4)
        
        # Decode distances to bounding boxes
        decoded_bboxes = dist2bbox(pred_dist, anchor_points, xywh=True)
        
        # Clamp decoded_bboxes to ensure they are within image boundaries
        decoded_bboxes = torch.clamp(decoded_bboxes, min=0.0, max=1e4)
        
        return decoded_bboxes



    def get_pred_boxes(self, pred):
        """Get predicted boxes for visualization."""
        pred = pred.detach()
        B, total_ch, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1).reshape(B, H * W, total_ch)
        
        dist_channels = self.reg_max * 4
        pred_dist = pred[..., :dist_channels]
        
        anchor_points = self.make_anchors(H, W, pred.device)
        return self.bbox_decode(anchor_points, pred_dist[0])

class QuaternionRegularizationLoss(nn.Module):
    """Regularization loss to enforce unit norm on quaternions."""
    
    def __init__(self, lambda_reg=1.0):
        """
        Initialize QuaternionRegularizationLoss.

        Args:
            lambda_reg (float): Weight for the regularization loss.
        """
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, pred_quat):
        """
        Compute the regularization loss.

        Args:
            pred_quat (torch.Tensor): Predicted quaternions, shape (..., 4)

        Returns:
            torch.Tensor: Regularization loss
        """
        norm = pred_quat.norm(p=2, dim=-1)
        loss = (norm - 1.0).pow(2).mean()
        return self.lambda_reg * loss

class OrientationSmoothnessLoss(nn.Module):
    """Loss to encourage smooth transitions in quaternion orientations."""
    
    def __init__(self, lambda_smooth=1.0):
        """
        Initialize OrientationSmoothnessLoss.

        Args:
            lambda_smooth (float): Weight for the smoothness loss.
        """
        super().__init__()
        self.lambda_smooth = lambda_smooth

    def forward(self, pred_quat, neighbor_quat):
        """
        Compute the orientation smoothness loss.

        Args:
            pred_quat (torch.Tensor): Predicted quaternions, shape (B, N, 4)
            neighbor_quat (torch.Tensor): Neighboring quaternions, shape (B, N, 4)

        Returns:
            torch.Tensor: Smoothness loss
        """
        # Ensure quaternions are normalized
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        neighbor_quat = F.normalize(neighbor_quat, p=2, dim=-1)

        # Compute angular difference
        dot_product = torch.abs((pred_quat * neighbor_quat).sum(dim=-1))
        dot_product = torch.clamp(dot_product, min=0.0, max=1.0)
        angular_diff = torch.acos(dot_product)  # radians

        # Penalize large differences
        loss = angular_diff.mean()
        return self.lambda_smooth * loss

class RotatedBBoxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated (quaternion-based) bounding boxes."""

    def __init__(self, reg_max=16, nc=80, lambda_reg=0.1, lambda_smooth=0.3):
        """
        Initialize the RotatedBBoxLoss module with regularization maximum and additional quaternion losses.

        Args:
            reg_max (int): Maximum value for distance bins.
            nc (int): Number of classes.
            lambda_reg (float): Weight for quaternion regularization loss.
            lambda_smooth (float): Weight for orientation smoothness loss.
        """
        super().__init__(reg_max, nc)
        self.quat_reg_loss = QuaternionRegularizationLoss(lambda_reg=lambda_reg).to('cuda' if torch.cuda.is_available() else 'cpu')
        self.orientation_smoothness_loss = OrientationSmoothnessLoss(lambda_smooth=lambda_smooth).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, pred_dist, target_bboxes, target_categories, pred_quat, target_quat):
        """
        Compute the rotated bounding box loss including quaternion regularization and smoothness.

        Args:
            pred_dist (torch.Tensor): Predicted distance distributions, shape (B*Q, num_anchors, 4*reg_max)
            target_bboxes (torch.Tensor): Target bounding boxes, shape (B, N, 4)
            target_categories (torch.Tensor): Target categories, shape (B, N)
            pred_quat (torch.Tensor): Predicted quaternions, shape (B*Q, num_anchors, 4)
            target_quat (torch.Tensor): Target quaternions, shape (B, N, 4)

        Returns:
            dict: Dictionary containing loss components and total loss
        """
        device = target_bboxes.device
        batch_size = target_bboxes.shape[0]
        
        # Process predictions
        if isinstance(pred_dist, (list, tuple)):
            pred_dist = pred_dist[0]  # Take first prediction level
        
        B, Q, num_anchors, _ = pred_dist.shape
        num_anchors = num_anchors  # Assuming num_anchors is already correct
        
        # Reshape predictions
        pred_dist = pred_dist.reshape(B * Q, num_anchors, -1)
        pred_quat = pred_quat.reshape(B * Q, num_anchors, -1)
        
        # Generate anchor points
        anchor_points = self.make_anchors(pred_dist.shape[-2], pred_dist.shape[-1], device)
        
        # Decode predictions to boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_dist)  # Shape: (B*Q, num_anchors, 4)
        
        box_loss = torch.tensor(0., device=device, requires_grad=True)
        cls_loss = torch.tensor(0., device=device, requires_grad=True)
        obj_loss = torch.tensor(0., device=device, requires_grad=True)
        quat_reg_loss = torch.tensor(0., device=device, requires_grad=True)
        smooth_loss = torch.tensor(0., device=device, requires_grad=True)
        
        num_valid_batches = 0
        
        for i in range(batch_size):
            # Get valid boxes for this batch
            valid_mask = target_bboxes[i].sum(dim=1) > 0
            if not valid_mask.any():
                continue
            
            valid_boxes = target_bboxes[i][valid_mask]
            valid_categories = target_categories[i][valid_mask]
            valid_quats = target_quat[i][valid_mask]
            
            # Match predictions to targets
            batch_pred_boxes = pred_bboxes[i * Q:(i + 1) * Q].reshape(-1, 4)  # Shape: (Q*num_anchors, 4)
            batch_pred_quat = pred_quat[i * Q:(i + 1) * Q].reshape(-1, 4)    # Shape: (Q*num_anchors, 4)
            
            # Compute IoU between predictions and targets
            ious = box_iou(batch_pred_boxes, valid_boxes, xywh=True)  # Shape: (Q*num_anchors, num_targets)
            
            # Find best IoU for each prediction
            best_ious, best_targets = ious.max(dim=1)  # Shape: (Q*num_anchors,)
            
            # Create classification targets
            batch_pred_scores = pred_scores[i * Q:(i + 1) * Q].reshape(-1, self.nc)  # Shape: (Q*num_anchors, nc)
            cls_targets = torch.zeros_like(batch_pred_scores)  # Shape: (Q*num_anchors, nc)
            
            positive_mask = best_ious > 0.5
            if positive_mask.any():
                matched_categories = valid_categories[best_targets[positive_mask]]
                cls_targets[positive_mask, matched_categories] = 1.0
                
                # Box loss - only for positive matches
                box_loss_i = (1 - best_ious[positive_mask]).mean()
                box_loss = box_loss + box_loss_i
                
                # Classification loss
                cls_loss_i = self.bce(batch_pred_scores, cls_targets).mean()
                cls_loss = cls_loss + cls_loss_i
                
                # Objectness loss
                obj_targets = (best_ious > 0.5).float()
                obj_loss_i = F.binary_cross_entropy_with_logits(best_ious, obj_targets)
                obj_loss = obj_loss + obj_loss_i
                
                # Quaternion Regularization Loss
                quat_pred = batch_pred_quat[positive_mask]
                quat_reg_loss_i = self.quat_reg_loss(quat_pred)
                quat_reg_loss = quat_reg_loss + quat_reg_loss_i
                
                # Orientation Smoothness Loss
                neighbor_quat = self.get_neighbor_quat(quat_pred)  # Implement this method based on your data
                smooth_loss_i = self.orientation_smoothness_loss(quat_pred, neighbor_quat)
                smooth_loss = smooth_loss + smooth_loss_i
                
                num_valid_batches += 1
        
        # Normalize losses by number of valid batches
        num_valid_batches = max(num_valid_batches, 1)
        box_loss = box_loss / num_valid_batches
        cls_loss = cls_loss / num_valid_batches
        obj_loss = obj_loss / num_valid_batches
        quat_reg_loss = quat_reg_loss / num_valid_batches
        smooth_loss = smooth_loss / num_valid_batches
        
        # Total loss with weightings
        total_loss = (box_loss * 3.0) + (cls_loss * 1.0) + (obj_loss * 1.0) + (quat_reg_loss * 0.1) + (smooth_loss * 0.3)
        
        return {
            'total_loss': total_loss,
            'box_loss': box_loss.detach(),
            'cls_loss': cls_loss.detach(),
            'obj_loss': obj_loss.detach(),
            'quat_reg_loss': quat_reg_loss.detach(),
            'smooth_loss': smooth_loss.detach()
        }

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predictions to boxes while maintaining gradients."""
        if self.use_dfl and pred_dist is not None:
            # Replace .view() with .reshape() to handle non-contiguous tensors
            pred_dist = pred_dist.reshape(-1, 4, self.reg_max).softmax(2)
            pred_dist = pred_dist @ torch.linspace(0, self.reg_max - 1, self.reg_max, device=pred_dist.device)
        return dist2bbox(pred_dist, anchor_points, xywh=True)
    
    @staticmethod
    def make_anchors(h, w, device):
        """Generate anchor points."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        
        grid_xy = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1)
        ], dim=1).float()
        
        # Add offset
        grid_xy += 0.5
        return grid_xy
    
    def get_neighbor_quat(self, pred_quat_fg):
        """Implement neighbor quaternion retrieval based on your specific data or model architecture."""
        # Example: Shift quaternions or use another strategy to define neighbors
        # Here, we'll use a simple roll operation as a placeholder
        neighbor_quat = torch.roll(pred_quat_fg, shifts=1, dims=0)
        return neighbor_quat


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses for keypoint regression."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area, pred_quat=None, target_quat=None):
        """
        Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints.
        Optionally includes quaternion-based orientation loss.

        Args:
            pred_kpts (torch.Tensor): Predicted keypoints, shape (B, num_keypoints, 2)
            gt_kpts (torch.Tensor): Ground truth keypoints, shape (B, num_keypoints, 2)
            kpt_mask (torch.Tensor): Mask indicating valid keypoints, shape (B, num_keypoints)
            area (torch.Tensor): Area of objects, shape (B, )
            pred_quat (torch.Tensor, optional): Predicted quaternions, shape (B, num_keypoints, 4)
            target_quat (torch.Tensor, optional): Target quaternions, shape (B, num_keypoints, 4)

        Returns:
            torch.Tensor: Combined keypoint and quaternion loss
        """
        # Euclidean distance loss for keypoints
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        loss = (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

        # Quaternion angular loss for keypoints (if applicable)
        if pred_quat is not None and target_quat is not None:
            # Ensure quaternions are normalized
            pred_quat = F.normalize(pred_quat, p=2, dim=-1)
            target_quat = F.normalize(target_quat, p=2, dim=-1)

            # Compute the absolute dot product
            dot_product = torch.abs((pred_quat * target_quat).sum(dim=-1))
            # Clamp for numerical stability
            dot_product = torch.clamp(dot_product, min=0.0, max=1.0)
            # Compute angular loss
            angular_loss = torch.acos(dot_product)  # radians
            # Weighted loss based on keypoint mask
            angular_loss = (angular_loss * kpt_mask).mean()
            loss += angular_loss

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    """
    Unified Detection Loss for standard and quaternion-based object detection.
    """
    def __init__(self, reg_max=16, nc=80, use_quat=False, tal_topk=10):
        """
        Initializes DetectionLoss with optional quaternion handling.

        Args:
            reg_max (int): Maximum value for distance bins.
            nc (int): Number of classes.
            use_quat (bool, optional): Flag to include quaternion-based losses. Defaults to False.
            tal_topk (int, optional): Top-K for Task-Aligned Assigners. Defaults to 10.
        """
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.use_quat = use_quat

        self.use_dfl = self.reg_max > 1

        # Initialize loss components
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(self.reg_max, self.nc)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, 
            num_classes=self.nc, 
            alpha=0.5, 
            beta=6.0
        )
        self.proj = torch.arange(self.reg_max, dtype=torch.float)

        # Quaternion-related losses
        if self.use_quat:
            self.quat_reg_loss = QuaternionRegularizationLoss(lambda_reg=0.1)
            self.geo_consistency_loss = GeometricConsistencyLoss(lambda_geo=0.5)
            self.orientation_smoothness_loss = OrientationSmoothnessLoss(lambda_smooth=0.3)


    def bbox_decode(self, pred_dist, anchor_points):
        """Delegate bbox_decode to BboxLoss instance."""
        return self.bbox_loss.bbox_decode(pred_dist, anchor_points)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, 
                target_scores_sum, fg_mask, pred_quat=None, target_quat=None):
        """
        Calculate all components of the detection loss.

        Args:
            pred_dist (torch.Tensor): Predicted distance distributions, shape (batch_size, num_anchors, 4 * reg_max)
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (batch_size, num_anchors, 4)
            anchor_points (torch.Tensor): Anchor points, shape (num_anchors, 2)
            target_bboxes (torch.Tensor): Ground truth bounding boxes, shape (batch_size, num_anchors, 4)
            target_scores (torch.Tensor): Objectness scores, shape (batch_size, num_anchors)
            target_scores_sum (torch.Tensor): Sum of objectness scores
            fg_mask (torch.Tensor): Foreground mask, shape (batch_size, num_anchors)
            pred_quat (torch.Tensor, optional): Predicted quaternions, shape (batch_size, num_anchors, 4)
            target_quat (torch.Tensor, optional): Ground truth quaternions, shape (batch_size, num_anchors, 4)

        Returns:
            tuple: (box_loss, dfl_loss, quat_loss)
        """

        # Compute Bounding Box Loss
        box_loss, dfl_loss, quat_loss = self.bbox_loss(
            pred_dist, 
            pred_bboxes, 
            anchor_points, 
            target_bboxes, 
            target_scores, 
            target_scores_sum, 
            fg_mask
        )

        # Initialize total loss
        total_loss = box_loss + dfl_loss

        # Compute Quaternion-related Losses if enabled
        if self.use_quat and pred_quat is not None and target_quat is not None:
            quat_reg = self.quat_reg_loss(pred_quat[fg_mask])
            geo_consistency = self.geo_consistency_loss(pred_bboxes[fg_mask], pred_quat[fg_mask])
            orientation_smoothness = self.orientation_smoothness_loss(
                pred_quat[fg_mask],
                self.get_neighbor_quat(pred_quat[fg_mask])
            )
            quat_loss = quat_reg + geo_consistency + orientation_smoothness
            total_loss += quat_loss

        return box_loss, dfl_loss, quat_loss

    @staticmethod
    def get_neighbor_quat(pred_quat):
        """Get neighboring quaternions for smoothness calculation."""
        # Simple implementation - can be enhanced based on your needs
        return torch.roll(pred_quat, shifts=1, dims=0)


    @staticmethod
    def make_anchors(h, w, device):
        """Generate anchor points."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        grid_xy = torch.stack([
            grid_x.reshape(-1),
            grid_y.reshape(-1)
        ], dim=1)
        
        # Add offset
        grid_xy += 0.5
        
        return grid_xy


    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

# class v8ClassificationLoss(nn.Module):
#     """Criterion class for computing training classification losses in quaternion-based models."""

#     def __init__(self, alpha=0.25, gamma=2.0):
#         """Initialize v8ClassificationLoss with focal loss parameters."""
#         super().__init__()
#         self.focal_loss = FocalLoss()
#         self.alpha = alpha
#         self.gamma = gamma

#     def __call__(self, preds, batch, pred_quat=None):
#         """Compute the classification loss between predictions and true labels."""
#         # Standard classification loss
#         loss = self.focal_loss(preds, batch["cls"], gamma=self.gamma, alpha=self.alpha)

#         # Optionally, integrate quaternion-based classification adjustments
#         if pred_quat is not None and batch.get("target_quat") is not None:
#             # Example: Penalize inconsistency between class predictions and quaternion orientations
#             # This requires defining how class and quaternion are related
#             quat_loss = self.quaternion_consistency_loss(pred_quat, batch["target_quat"])
#             loss += quat_loss

#         loss_items = loss.detach()
#         return loss, loss_items

#     @staticmethod
#     def quaternion_consistency_loss(pred_quat, target_quat):
#         """
#         Encourage consistency between class predictions and quaternion orientations.
#         For example, certain classes might have preferred orientation ranges.

#         Args:
#             pred_quat (torch.Tensor): Predicted quaternions, shape (B, C, 4)
#             target_quat (torch.Tensor): Target quaternions, shape (B, C, 4)

#         Returns:
#             torch.Tensor: Quaternion consistency loss
#         """
#         # Example: Use angular loss between predicted and target quaternions
#         pred_quat = F.normalize(pred_quat, p=2, dim=-1)
#         target_quat = F.normalize(target_quat, p=2, dim=-1)

#         dot_product = torch.abs((pred_quat * target_quat).sum(dim=-1))
#         dot_product = torch.clamp(dot_product, min=0.0, max=1.0)
#         angular_loss = torch.acos(dot_product)  # radians

#         return angular_loss.mean()

# # Modified v8OBBLoss with novel enhancements
# class v8OBBLoss(v8DetectionLoss):
#     """Calculates losses for quaternion-based object detection, including rotated bounding boxes."""

#     def __init__(self, model):
#         """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
#         super().__init__(model)
#         self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
#         self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
#         self.quat_reg_loss = QuaternionRegularizationLoss(lambda_reg=0.1).to(self.device)
#         self.geo_consistency_loss = GeometricConsistencyLoss(lambda_geo=0.5).to(self.device)
#         self.orientation_smoothness_loss = OrientationSmoothnessLoss(lambda_smooth=0.3).to(self.device)

#     def __call__(self, preds, batch):
#         """Calculate and return the loss for the quaternion-based YOLO model."""
#         loss = torch.zeros(4, device=self.device)  # box, cls, dfl, quat
#         feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
#         pred_distri, pred_scores, pred_quat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc, self.no - self.reg_max * 4 - self.nc), 1
#         )

#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()
#         pred_quat = pred_quat.permute(0, 2, 1).contiguous()

#         dtype = pred_scores.dtype
#         batch_size = pred_scores.shape[0]
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

#         # Targets
#         try:
#             batch_idx = batch["batch_idx"].view(-1, 1)
#             targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 6)), 1)  # [batch_idx, cls, x, y, w, h, quat]
#             # Filter out tiny rotated boxes
#             rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
#             targets = targets[(rw >= 2) & (rh >= 2)]  # filter rotated boxes of tiny size to stabilize training
#             targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#             gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, [x, y, w, h, quat]
#             mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
#         except RuntimeError as e:
#             raise TypeError(
#                 "ERROR ❌ OBB dataset incorrectly formatted or not an OBB dataset.\n"
#                 "This error can occur when incorrectly training an 'OBB' model on a 'detect' dataset, "
#                 "i.e., 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
#                 "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
#                 "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
#             ) from e

#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_quat)  # [x, y, w, h, quat], (b, h*w, 8)

#         # Assign targets
#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             pred_scores.detach().sigmoid(),
#             pred_bboxes.detach()[:, :, :4].type(gt_bboxes.dtype),  # only spatial for assignment
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes[..., :4],
#             mask_gt,
#         )

#         target_scores_sum = max(target_scores.sum(), 1)

#         # Cls loss
#         loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

#         # Bbox loss
#         if fg_mask.sum():
#             target_bboxes /= stride_tensor
#             # Extract quaternion components
#             target_quat = gt_bboxes[fg_mask, 4:8]
#             pred_quat_fg = pred_bboxes[fg_mask, 4:8]
#             target_bboxes = target_bboxes[fg_mask, :4]
#             loss_iou, loss_dfl, loss_quat = self.bbox_loss(
#                 pred_distri, pred_bboxes[fg_mask, :4], anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, pred_quat_fg, target_quat
#             )
#             loss_quat += self.quat_reg_loss(pred_quat_fg)
#             # Geometric consistency loss
#             loss_geo = self.geo_consistency_loss(pred_bboxes[fg_mask, :4], pred_quat_fg)
#             # Orientation smoothness loss
#             neighbor_quat = self.get_neighbor_quat(pred_quat_fg)  # Implement this method based on your data
#             loss_smooth = self.orientation_smoothness_loss(pred_quat_fg, neighbor_quat)
#             # Aggregate losses
#             loss[0] += loss_iou * self.hyp.box
#             loss[2] += loss_dfl * self.hyp.dfl
#             loss[3] += (loss_quat * self.hyp.quat + loss_geo + loss_smooth)
#         else:
#             loss[0] += 0
#             loss[2] += 0
#             loss[3] += 0

#         return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, quat)

class E2EDetectLoss:
    """Criterion class for computing end-to-end training losses for quaternion-based detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one quaternion-based detection losses."""
        self.one2many = v8OBBLoss(model)  # Use v8OBBLoss for one-to-many path
        self.one2one = v8OBBLoss(model)   # Use v8OBBLoss for one-to-one path

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls, dfl, and quat for both detection paths."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many, _ = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one, _ = self.one2one(one2one, batch)
        return loss_one2many + loss_one2one, (loss_one2many, loss_one2one)

class GeometricConsistencyLoss(nn.Module):
    """Loss to ensure consistency between bounding box geometry and quaternion orientations."""

    def __init__(self, lambda_geo=1.0):
        """
        Initialize GeometricConsistencyLoss.

        Args:
            lambda_geo (float): Weight for the geometric consistency loss.
        """
        super().__init__()
        self.lambda_geo = lambda_geo

    def forward(self, pred_bboxes, pred_quat):
        """
        Compute the geometric consistency loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes, shape (..., 4)
            pred_quat (torch.Tensor): Predicted quaternions, shape (..., 4)

        Returns:
            torch.Tensor: Geometric consistency loss
        """
        # Example: Align quaternion orientation with bounding box aspect ratio
        # Compute aspect ratio
        w, h = pred_bboxes[..., 2], pred_bboxes[..., 3]
        aspect_ratio = w / (h + 1e-8)
        # Compute expected quaternion based on aspect ratio (simplistic example)
        # Define a mapping from aspect ratio to quaternion
        expected_quat = self.map_aspect_to_quaternion(aspect_ratio)
        # Compute angular loss between predicted and expected quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        expected_quat = F.normalize(expected_quat, p=2, dim=-1)
        dot_product = torch.abs((pred_quat * expected_quat).sum(dim=-1))
        dot_product = torch.clamp(dot_product, min=0.0, max=1.0)
        angular_loss = torch.acos(dot_product)  # radians
        return self.lambda_geo * angular_loss.mean()

    @staticmethod
    def map_aspect_to_quaternion(aspect_ratio):
        """
        Map aspect ratio to expected quaternion. This is a placeholder and should be defined based on application.

        Args:
            aspect_ratio (torch.Tensor): Aspect ratio, shape (..., )

        Returns:
            torch.Tensor: Expected quaternions, shape (..., 4)
        """
        # Example: For simplicity, assume rotation around the z-axis based on aspect ratio
        # More sophisticated mappings can be implemented based on application needs
        angles = torch.atan(aspect_ratio)
        half_angles = angles / 2
        quat = torch.stack([
            torch.zeros_like(half_angles),
            torch.zeros_like(half_angles),
            torch.sin(half_angles),
            torch.cos(half_angles)
        ], dim=-1)
        return quat
