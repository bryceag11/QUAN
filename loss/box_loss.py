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

class DetectionLoss(nn.Module):
    def __init__(self, reg_max=9, nc=80, use_dfl=True):  # Changed reg_max from 16 to 9
        super().__init__()
        self.reg_max = reg_max
        self.nc = nc
        self.use_dfl = use_dfl
        self.dfl = DFLoss(reg_max) if use_dfl else nn.Identity()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.iou_loss = nn.SmoothL1Loss(reduction='none')
        self.quaternion_loss = QuaternionLoss()
        
    def forward(self, outputs, anchor_points, target_bboxes, target_categories, target_scores_sum):
        """
        Args:
            outputs: List of outputs from QDetectHead, each with shape [B, 36, Q, H, W]
                    where 36 = 4 * reg_max(9)
            anchor_points: Dict of anchor points for each level
            target_bboxes: Shape [B, M, 4]
            target_categories: Shape [B, M]
            target_scores_sum: Scalar tensor
        """
        total_box_loss = 0
        total_dfl_loss = 0 
        total_cls_loss = 0
        total_quat_loss = 0
        num_levels = len(outputs)

        print("\nProcessing detection loss:")
        for i, pred in enumerate(outputs):
            print(f"Level {i} output shape: {pred.shape}")
        print(f"Target bboxes shape: {target_bboxes.shape}")
        print(f"Target categories shape: {target_categories.shape}")
        
        for level_idx, pred in enumerate(outputs):
            # Get dimensions
            B, C, Q, H, W = pred.shape
            print(f"\nLevel {level_idx}:")
            print(f"B={B}, C={C}, Q={Q}, H={H}, W={W}")

            # Split channels - expect C=36 total channels:
            # - First 36 channels: 4 coordinates * 9 reg_max values = 36 for regression
            reg_channels = 4 * self.reg_max  # Should be 36
            
            # Reshape predictions - preserve batch dim for clarity
            pred = pred.permute(0, 3, 4, 2, 1)  # [B, H, W, Q, C]
            pred = pred.reshape(B, H * W, Q, C)  # [B, H*W, Q, C]
            
            # Split into regression only since all channels are for regression
            pred_reg = pred  # [B, H*W, Q, C]

            # Get anchor points for this level
            level_anchors = anchor_points[level_idx]  # [H*W, 2]
            
            if self.use_dfl:
                # Reshape for DFL - maintain batch dimension
                pred_dist = pred_reg.reshape(B, H*W, Q, 4, self.reg_max)
                pred_dist = F.softmax(pred_dist, dim=-1)
                
                # Process each batch independently
                pred_boxes = []
                for b in range(B):
                    # [H*W, Q, 4, reg_max] -> [H*W*Q, 4, reg_max]
                    batch_dist = pred_dist[b].reshape(-1, 4,reg_channels)
                    # Get distribution-based coordinates
                    batch_boxes = dist2bbox(
                        batch_dist.reshape(-1,reg_channels), 
                        level_anchors.repeat_interleave(Q, dim=0)
                    )  # [H*W*Q, 4]
                    pred_boxes.append(batch_boxes)
                pred_boxes = torch.stack(pred_boxes)  # [B, H*W*Q, 4]
            else:
                pred_boxes = dist2bbox(pred_reg.reshape(B*H*W*Q, -1), level_anchors)
                pred_boxes = pred_boxes.reshape(B, H*W*Q, 4)

            # Compute losses for boxes and quaternions
            target_boxes_level = target_bboxes.reshape(B, -1, 4)  # [B, M, 4]
            
            # Box IoU loss
            iou = bbox_iou(
                pred_boxes.reshape(-1, 4),
                target_boxes_level.reshape(-1, 4),
                xywh=True
            )
            box_loss = -torch.log(iou + 1e-8).mean()
            
            if self.use_dfl:
                # Prepare anchor points 
                expanded_anchors = level_anchors.unsqueeze(0).expand(B, -1, -1)  # [B, H*W, 2]
                
                # Get target distributions
                target_dist = bbox2dist(
                    expanded_anchors,  # [B, H*W, 2]
                    target_boxes_level,  # [B, M, 4] 
                    reg_max=self.reg_max
                )  # [B, H*W, 4*reg_max]
                target_dist = target_dist.reshape(B, H*W, 4, self.reg_max)  # [B, H*W, 4, reg_max]
                target_dist = target_dist.unsqueeze(2)  # [B, H*W, 1, 4, reg_max]
                target_dist = target_dist.expand(-1, -1, Q, -1, -1)  # [B, H*W, Q, 4, reg_max]

                # Prepare predictions - reshape to match targets
                pred_dist = pred_dist.reshape(B, H*W*Q, 4, self.reg_max)
                
                # Compute DFL loss
                flat_pred = pred_dist.reshape(-1, self.reg_max)  # [B*H*W*Q*4, reg_max]
                flat_target = target_dist.reshape(-1, self.reg_max)  # [B*H*W*Q*4, reg_max]

                dfl_loss = self.dfl(flat_pred, flat_target)
            else:
                dfl_loss = torch.tensor(0.0, device=pred_reg.device)


            # Quaternion loss
            quat_loss = self.quaternion_loss(
                pred_boxes.reshape(B, -1, Q, 4)  # Reshape to expose quaternion dimension
            )

            # Accumulate losses
            total_box_loss += box_loss 
            total_dfl_loss += dfl_loss
            total_quat_loss += quat_loss

            print(f"Level {level_idx} losses:")
            print(f"Box loss: {box_loss.item():.4f}")
            print(f"DFL loss: {dfl_loss.item():.4f}") 
            print(f"Quat loss: {quat_loss.item():.4f}")

        # Average across levels
        num_valid_levels = num_levels
        total_box_loss = total_box_loss / num_valid_levels
        total_dfl_loss = total_dfl_loss / num_valid_levels
        total_quat_loss = total_quat_loss / num_valid_levels

        print("\nFinal losses:")
        print(f"Total box loss: {total_box_loss.item():.4f}")
        print(f"Total DFL loss: {total_dfl_loss.item():.4f}")
        print(f"Total quat loss: {total_quat_loss.item():.4f}")
        
        return total_box_loss, total_dfl_loss, 0.0, total_quat_loss  # Return 0 for cls_loss since no classification


class QuaternionLoss(nn.Module):
    """
    Specialized loss for quaternion predictions in object detection.
    Handles both rotation error and quaternion constraints.
    """
    def __init__(self, quaternion_weight=1.0):
        super().__init__()
        self.quaternion_weight = quaternion_weight
        
    def forward(self, pred_quat, target_quat):
        """
        Args:
            pred_quat: Predicted quaternions [B, N, 4]
            target_quat: Target quaternions [B, N, 4]
        """
        # Normalize quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        target_quat = F.normalize(target_quat, p=2, dim=-1)
        
        # Double cover handling: q and -q represent same rotation
        dot_abs = torch.abs(torch.sum(pred_quat * target_quat, dim=-1))
        dot_abs = torch.clamp(dot_abs, min=0.0, max=1.0)
        
        # Angular loss
        angle_loss = 2 * torch.acos(dot_abs)
        
        # Unit norm constraint
        norm_loss = torch.abs(torch.sum(pred_quat * pred_quat, dim=-1) - 1.0)
        
        # Combine losses
        total_loss = angle_loss.mean() + 0.1 * norm_loss.mean()
        
        return self.quaternion_weight * total_loss

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
    def __init__(self, reg_max=9):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, target):
        # target: [N] with values in [0, reg_max - 1]
        # pred_dist: [N, reg_max], raw logits (no softmax)
        
        target = target.clamp(0, self.reg_max - 1)  # [N]
        target_float = target.float()
        
        left = target  # [N], long
        right = (left + 1).clamp(max=self.reg_max - 1)  # [N], long

        weight_right = target_float - left.float()
        weight_left = 1.0 - weight_right

        # pred_dist should be [N, reg_max] raw logits
        loss_left = F.cross_entropy(pred_dist, left, reduction='none')
        loss_right = F.cross_entropy(pred_dist, right, reduction='none')

        return (weight_left * loss_left + weight_right * loss_right).mean()





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
    def __init__(self, reg_max=16, nc=80, use_quat=True):
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

    # in trainer.py
    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        
        # Verify model is in training mode
        print(f"\nStarting epoch {epoch}")
        print(f"Model training mode: {self.model.training}")
        trainable_params = sum(p.requires_grad for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params}")
        
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        
        epoch_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'dfl_loss': 0.0,
            'quat_loss': 0.0
        }
        
        for batch_idx, batch in progress_bar:
            try:
                # Clear memory and gradients
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                
                # Check batch validity
                if batch['bbox'].numel() == 0:
                    print(f"\nSkipping empty batch {batch_idx}")
                    continue
                
                # Print batch stats
                print(f"\nBatch {batch_idx} content:")
                print(f"Images: {batch['image'].shape}")
                print(f"Boxes: {batch['bbox'].shape}")
                print(f"Categories: {batch['category'].shape}")
                print(f"Valid boxes: {(batch['bbox'].sum(dim=-1) != 0).sum()}")
                
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                target_bboxes = batch['bbox'].to(self.device, non_blocking=True)
                target_categories = batch['category'].to(self.device, non_blocking=True)
                
                # Memory check after data loading
                print(f"\n[Batch {batch_idx}] After data loading:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e6:.2f} MB")
                
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    # Get predictions
                    preds = self.model(images)
                    pred = preds[0] if isinstance(preds, (list, tuple)) else preds
                    
                    # Validate predictions
                    if pred is None or pred.numel() == 0:
                        print(f"Empty predictions in batch {batch_idx}")
                        continue
                        
                    # Check prediction stats
                    print(f"\nPrediction stats:")
                    print(f"Shape: {pred.shape}")
                    print(f"Mean: {pred.mean().item():.4f}")
                    print(f"Std: {pred.std().item():.4f}")
                    print(f"Requires grad: {pred.requires_grad}")
                    
                    # Prepare predictions for loss
                    B, C, D, H, W = pred.shape
                    num_anchors = H * W
                    pred = pred.permute(0, 3, 4, 1, 2).reshape(B, num_anchors, -1)
                    reg_max = 16
                    pred_dist = pred[..., :reg_max * 4]
                    
                    # Generate anchors
                    anchor_points = self.loss_fn.make_anchors(H, W, pred.device)
                    pred_bboxes = self.loss_fn.bbox_decode(pred_dist, anchor_points)
                    
                    # Prepare targets
                    fg_mask = torch.zeros(B, num_anchors, dtype=torch.bool, device=self.device)
                    target_scores = torch.zeros(B, num_anchors, device=self.device)
                    
                    # Find valid targets
                    valid_targets = 0
                    for b in range(B):
                        valid_mask = target_categories[b] > 0
                        num_valid = valid_mask.sum().item()
                        valid_targets += num_valid
                        if num_valid > 0:
                            fg_mask[b, :num_valid] = True
                            target_scores[b, :num_valid] = 1.0
                    
                    print(f"Valid targets: {valid_targets}")
                    
                    if valid_targets == 0:
                        print(f"No valid targets in batch {batch_idx}")
                        continue
                    
                    target_scores_sum = target_scores.sum().clamp(min=1)
                    
                    # Compute losses
                    box_loss, dfl_loss, quat_loss = self.loss_fn(
                        pred_dist=pred_dist,
                        pred_bboxes=pred_bboxes,
                        anchor_points=anchor_points,
                        target_bboxes=target_bboxes,
                        target_scores=target_scores,
                        target_scores_sum=target_scores_sum,
                        fg_mask=fg_mask
                    )
                    
                    # Validate losses
                    total_loss = box_loss + dfl_loss
                    if quat_loss is not None:
                        total_loss += quat_loss
                    
                    if not total_loss.requires_grad:
                        print(f"Warning: total_loss lost gradients")
                        print(f"box_loss requires_grad: {box_loss.requires_grad}")
                        print(f"dfl_loss requires_grad: {dfl_loss.requires_grad}")
                        if quat_loss is not None:
                            print(f"quat_loss requires_grad: {quat_loss.requires_grad}")
                        raise ValueError("Loss doesn't require gradients")
                
                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.grad_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                
                # Optimize
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update metrics
                with torch.no_grad():
                    epoch_metrics['total_loss'] += total_loss.item()
                    epoch_metrics['box_loss'] += box_loss.item()
                    epoch_metrics['dfl_loss'] += dfl_loss.item()
                    if quat_loss is not None:
                        epoch_metrics['quat_loss'] += quat_loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'box': f"{box_loss.item():.4f}",
                    'dfl': f"{dfl_loss.item():.4f}",
                    'quat': f"{quat_loss.item() if quat_loss is not None else 0.0:.4f}"
                })
                
                # Memory check after optimization
                print(f"\n[Batch {batch_idx}] After optimizer step:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e6:.2f} MB")
                
            except RuntimeError as e:
                print(f"\nError in batch {batch_idx}:")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                print("\nLast tensor states:")
                print(f"pred_dist shape: {pred_dist.shape}")
                print(f"pred_bboxes shape: {pred_bboxes.shape}")
                print(f"box_loss: {box_loss}")
                print(f"dfl_loss: {dfl_loss}")
                print(f"total_loss: {total_loss}")
                print("\nCUDA Memory Status:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                raise e
            
            finally:
                # Cleanup
                torch.cuda.empty_cache()

        # Calculate epoch metrics
        num_batches = len(self.train_dataloader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
        return epoch_metrics

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