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
    """Criterion class for computing Distribution Focal Loss (DFL) for quaternion-based bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target, pred_quat=None, target_quat=None):
        """
        Return sum of left and right DFL losses.
        For OBB, handle quaternion distribution if necessary.
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right

        loss = (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)

        # Optionally, add quaternion-specific loss terms here
        if self.reg_max > 4 and pred_quat is not None and target_quat is not None:
            # Example: Quaternion angular loss (e.g., geodesic distance)
            quat_loss = self.quaternion_angular_loss(pred_quat, target_quat)
            loss += quat_loss

        return loss

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
    """Criterion class for computing bounding box regression losses, including quaternion-based orientation."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, pred_quat=None, target_quat=None):
        """IoU and quaternion-aware regression loss."""
        # IoU loss (e.g., CIoU)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss for bounding box regression
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # Quaternion angular loss
        if pred_quat is not None and target_quat is not None and fg_mask.sum():
            loss_quat = self.quaternion_angular_loss(pred_quat[fg_mask], target_quat[fg_mask]) * weight
            loss_quat = loss_quat.sum() / target_scores_sum
        else:
            loss_quat = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, loss_quat

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



class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated (quaternion-based) bounding boxes."""

    def __init__(self, reg_max):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, pred_quat, target_quat):
        """IoU and quaternion-aware regression loss for rotated bounding boxes."""
        # IoU loss (e.g., Proportional IoU for rotated boxes)
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])  # Assuming probiou handles rotated IoU
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss for bounding box regression
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # Quaternion angular loss
        if pred_quat is not None and target_quat is not None and fg_mask.sum():
            loss_quat = self.quaternion_angular_loss(pred_quat[fg_mask], target_quat[fg_mask]) * weight
            loss_quat = loss_quat.sum() / target_scores_sum
        else:
            loss_quat = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, loss_quat



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


class v8DetectionLoss(nn.Module):
    """Criterion class for computing training losses for quaternion-based object detection."""

    def __init__(self, model, tal_topk=10):
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        super().__init__()
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4 + m.ne  # Adjusted for quaternion orientation
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

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

    def bbox_decode(self, anchor_points, pred_dist, pred_quat=None):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # Decode bounding boxes
        bboxes = dist2bbox(pred_dist, anchor_points, xywh=False)
        if pred_quat is not None:
            # Decode quaternion from predicted distributions or angles
            # Assuming pred_quat is directly predicted as quaternion components
            bboxes = torch.cat((bboxes, F.normalize(pred_quat, p=2, dim=-1)), dim=-1)  # Append quaternion
        return bboxes

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls, dfl, and quat multiplied by batch size."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, quat
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores, pred_quat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, self.no - self.reg_max * 4 - self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_quat = pred_quat.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_quat)  # xyxy + quaternion, (b, h*w, 8)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach()[:, :, :4].type(gt_bboxes.dtype),  # only spatial for assignment
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # Extract quaternion components
            target_quat = target_bboxes[..., 4:8]
            pred_quat_fg = pred_bboxes[fg_mask, 4:8]
            target_bboxes = target_bboxes[..., :4]
            loss[0], loss[2], loss[3] = self.bbox_loss(
                pred_distri, pred_bboxes[..., :4], anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, pred_quat_fg, target_quat
            )
        else:
            loss[0] += (pred_quat * 0).sum()
            loss[2] += (pred_distri * 0).sum()
            loss[3] += (pred_quat * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.quat  # quat gain, ensure 'quat' is defined in hyperparameters

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, quat)

class v8ClassificationLoss(nn.Module):
    """Criterion class for computing training classification losses in quaternion-based models."""

    def __init__(self, alpha=0.25, gamma=2.0):
        """Initialize v8ClassificationLoss with focal loss parameters."""
        super().__init__()
        self.focal_loss = FocalLoss()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, preds, batch, pred_quat=None):
        """Compute the classification loss between predictions and true labels."""
        # Standard classification loss
        loss = self.focal_loss(preds, batch["cls"], gamma=self.gamma, alpha=self.alpha)

        # Optionally, integrate quaternion-based classification adjustments
        if pred_quat is not None and batch.get("target_quat") is not None:
            # Example: Penalize inconsistency between class predictions and quaternion orientations
            # This requires defining how class and quaternion are related
            quat_loss = self.quaternion_consistency_loss(pred_quat, batch["target_quat"])
            loss += quat_loss

        loss_items = loss.detach()
        return loss, loss_items

    @staticmethod
    def quaternion_consistency_loss(pred_quat, target_quat):
        """
        Encourage consistency between class predictions and quaternion orientations.
        For example, certain classes might have preferred orientation ranges.

        Args:
            pred_quat (torch.Tensor): Predicted quaternions, shape (B, C, 4)
            target_quat (torch.Tensor): Target quaternions, shape (B, C, 4)

        Returns:
            torch.Tensor: Quaternion consistency loss
        """
        # Example: Use angular loss between predicted and target quaternions
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        target_quat = F.normalize(target_quat, p=2, dim=-1)

        dot_product = torch.abs((pred_quat * target_quat).sum(dim=-1))
        dot_product = torch.clamp(dot_product, min=0.0, max=1.0)
        angular_loss = torch.acos(dot_product)  # radians

        return angular_loss.mean()

# Modified v8OBBLoss with novel enhancements
class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for quaternion-based object detection, including rotated bounding boxes."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
        self.quat_reg_loss = QuaternionRegularizationLoss(lambda_reg=0.1).to(self.device)
        self.geo_consistency_loss = GeometricConsistencyLoss(lambda_geo=0.5).to(self.device)
        self.orientation_smoothness_loss = OrientationSmoothnessLoss(lambda_smooth=0.3).to(self.device)

    def __call__(self, preds, batch):
        """Calculate and return the loss for the quaternion-based YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, quat
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores, pred_quat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, self.no - self.reg_max * 4 - self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_quat = pred_quat.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 6)), 1)  # [batch_idx, cls, x, y, w, h, quat]
            # Filter out tiny rotated boxes
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rotated boxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, [x, y, w, h, quat]
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not an OBB dataset.\n"
                "This error can occur when incorrectly training an 'OBB' model on a 'detect' dataset, "
                "i.e., 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_quat)  # [x, y, w, h, quat], (b, h*w, 8)

        # Assign targets
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes.detach()[:, :, :4].type(gt_bboxes.dtype),  # only spatial for assignment
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes[..., :4],
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # Extract quaternion components
            target_quat = gt_bboxes[fg_mask, 4:8]
            pred_quat_fg = pred_bboxes[fg_mask, 4:8]
            target_bboxes = target_bboxes[fg_mask, :4]
            loss_iou, loss_dfl, loss_quat = self.bbox_loss(
                pred_distri, pred_bboxes[fg_mask, :4], anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, pred_quat_fg, target_quat
            )
            loss_quat += self.quat_reg_loss(pred_quat_fg)
            # Geometric consistency loss
            loss_geo = self.geo_consistency_loss(pred_bboxes[fg_mask, :4], pred_quat_fg)
            # Orientation smoothness loss
            neighbor_quat = self.get_neighbor_quat(pred_quat_fg)  # Implement this method based on your data
            loss_smooth = self.orientation_smoothness_loss(pred_quat_fg, neighbor_quat)
            # Aggregate losses
            loss[0] += loss_iou * self.hyp.box
            loss[2] += loss_dfl * self.hyp.dfl
            loss[3] += (loss_quat * self.hyp.quat + loss_geo + loss_smooth)
        else:
            loss[0] += 0
            loss[2] += 0
            loss[3] += 0

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl, quat)

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
        loss = (angular_diff).mean()
        return self.lambda_smooth * loss

