# models/heads/qdet_head.py

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from utils.ops import dist2bbox, dist2rbox, make_anchors
from quaternion.conv import QConv2D, DWConv, Conv, QDense  # Import Quaternion-aware Conv layer
from quaternion.qactivation import QReLU  # Import Quaternion-aware activation
from quaternion.qbatch_norm import QBN  # Import Quaternion-aware BatchNorm
from typing import List, Union
from quaternion.qbatch_norm import IQBN
import math 
from typing import List, Tuple
import torch.nn.functional as F
from utils.tal import TaskAlignedAssigner
from utils.ops import make_anchors, dist2bbox
from models.blocks import DFL



class QuaternionPooling(nn.Module):
    """Handle spatial pooling for 5D quaternion tensors."""
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W)
        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, 4, H_out, W_out)
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)
        
        # Apply pooling
        pooled = self.pool(x_reshaped)
        
        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

class QExtractReal(nn.Module):
    """
    Expands quaternion channels by extracting & weighting each component properly.
    Input: [B, C, 4, H, W] -> Output: [B, 4*C, H, W]
    """
    def __init__(self):
        super().__init__()
        # Initialize learnable weights for quaternion components
        self.weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32))
    
    def forward(self, x):
        """
        Forward pass that properly expands quaternion channels
        Args:
            x: [B, C, 4, H, W] quaternion tensor
        Returns:
            [B, 4*C, H, W] expanded real tensor
        """
        # x shape: [B, C, 4, H, W]
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components"
        
        # Get normalized weights - keep in FP32 for stability
        weights = F.softmax(self.weights, dim=0).to(device=x.device)
        
        # Extract and weight each quaternion component separately
        components = []
        for i in range(4):
            # Extract this component and weight it
            comp = x[:, :, i, :, :] * weights[i]  # [B, C, H, W]
            components.append(comp)
            
        # Concatenate all weighted components along channel dimension
        x = torch.cat(components, dim=1)  # [B, 4*C, H, W]
        
        return x

class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),  # Using c2 for intermediate channels
                Conv(c2, c2, 3), 
                # QExtractReal(),  # Extract real component and scale back
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),  # Using c3 for intermediate channels
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                # QExtractReal(),  # Extract real component and scale back
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    # def forward(self, x):
    #     """Concatenates and returns predicted bounding boxes and class probabilities."""
    #     if self.end2end:
    #         return self.forward_end2end(x)

    #     for i in range(self.nl):
    #         x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
    #     if self.training:  # Training path
    #         return x
    #     y = self._inference(x)
    #     return y if self.export else (y, x)

    # def forward(self, x):
    #     """Concatenates and returns predicted bounding boxes and class probabilities."""
    #     if self.end2end:
    #         return self.forward_end2end(x)
    #     dtype = x[0].dtype
    #     device = x[0].device

    #     for i in range(self.nl):
    #         # Convert modules to correct dtype and device
    #         self.cv2[i] = self.cv2[i].to(device)
    #         self.cv3[i] = self.cv3[i].to(device)
    #         x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1).to(device)


    #     if self.training:  # Training path
    #         return x
    #     y = self._inference(x)
    #     return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        dtype = x[0].dtype  # Get dtype from input

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2) #.to(dtype=dtype)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


# class QDetectHead(nn.Module):
#     def __init__(self, nc=80, ch=()):
#         super().__init__()
#         self.nc = nc
#         self.nl = len(ch)  
#         self.stride = torch.zeros(self.nl)
#         self.ch = ch

#         # Classification heads (real-valued) and quaternion regression heads
#         self.cls_heads = nn.ModuleList()
#         self.reg_heads = nn.ModuleList()
        
#         # Build heads for each feature level
#         for c in ch:
#             # Classification head - operates on real component only
#             cls_head = nn.Sequential(
#                 nn.Conv2d(c//4, c//8, 3, padding=1),  # c//4 because we take real component
#                 nn.BatchNorm2d(c//8),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(c//8, self.nc, 1)
#             )
#             self.cls_heads.append(cls_head)
            
#             # Box regression head - quaternion-aware for rotation info
#             reg_head = nn.Sequential(
#                 QConv2D(c, c//2, 3, padding=1),  # Takes full quaternion input
#                 IQBN(c//2),
#                 QReLU(),
#                 QConv2D(c//2, 16, 1)  # 4 box params × 4 quaternion components
#             )
#             self.reg_heads.append(reg_head)

#     def make_anchors(self, feats):
#         """
#         Generate anchor points for quaternion feature maps.
#         feats shape: List of [B, C, 4, H, W]
#         """
#         anchor_points, stride_tensor = [], []
        
#         for i, feat in enumerate(feats):
#             _, _, _, h, w = feat.shape  # Note the 5D shape
#             sx = torch.arange(w, device=feat.device) + 0.5
#             sy = torch.arange(h, device=feat.device) + 0.5
#             sy, sx = torch.meshgrid(sy, sx, indexing='ij')
#             anchor_points.append(torch.stack((sx, sy), -1).reshape(-1, 2) * self.stride[i])
#             stride_tensor.append(torch.full((h * w,), self.stride[i], device=feat.device))
            
#         return torch.cat(anchor_points), torch.cat(stride_tensor)

#     def forward(self, x):
#         """Forward pass handling quaternion format"""
#         cls_outputs = []
#         reg_outputs = []
        
#         # Generate anchors from quaternion feature maps
#         anchor_points, stride_tensor = self.make_anchors(x)
        
#         for i, feat in enumerate(x):
#             B, C, Q, H, W = feat.shape
            
#             # Classification: use real component only
#             feat_cls = feat[:, :, 0]  # [B, C, H, W]
#             cls_out = self.cls_heads[i](feat_cls)
            
#             # Regression: use full quaternion tensor
#             reg_out = self.reg_heads[i](feat)  # [B, 16, 4, H, W]
#             reg_out = self._combine_quaternion_components(reg_out)  # [B, 4, H, W]
            
#             # Reshape outputs
#             cls_out = cls_out.view(B, self.nc, -1).permute(0, 2, 1)  # [B, H*W, nc]
#             reg_out = reg_out.view(B, 4, -1).permute(0, 2, 1)  # [B, H*W, 4]
            
#             cls_outputs.append(cls_out)
#             reg_outputs.append(reg_out)
            
#         return cls_outputs, reg_outputs, anchor_points
    
#     def _combine_quaternion_components(self, quat_output):
#         """Simplified quaternion combination focusing on stable predictions"""
#         B, C, Q, H, W = quat_output.shape
#         assert C == 4 and Q == 4
        
#         # Extract components but use simpler weighting
#         r = quat_output[:, :, 0]  # Real component
#         i = quat_output[:, :, 1]  # First imaginary
#         j = quat_output[:, :, 2]  # Second imaginary 
#         k = quat_output[:, :, 3]  # Third imaginary
        
#         # Weight real component more heavily
#         combined = (2*r + i + j + k) / 5.0  # Bias toward real component
        
#         return combined
        
#     def get_targets(self, imgs, targets):
#         """Assign targets using TaskAlignedAssigner."""
#         batch_size = len(imgs)
#         device = targets['boxes'].device
        
#         # Make anchors for all feature levels
#         anchors = []
#         for feat in imgs:
#             anchor = make_anchors(feat, self.stride)
#             anchors.append(anchor)
        
#         anchors = torch.cat(anchors, dim=0)
        
#         # Initialize targets
#         target_labels = []
#         target_bboxes = []
#         target_masks = []
        
#         # Process each image in batch
#         for i in range(batch_size):
#             gt_boxes = targets['boxes'][i]
#             gt_labels = targets['labels'][i]
            
#             # Skip if no ground truth
#             if len(gt_boxes) == 0:
#                 target_labels.append(torch.zeros((0, self.nc), device=device))
#                 target_bboxes.append(torch.zeros((0, 4), device=device))
#                 target_masks.append(torch.zeros(0, dtype=torch.bool, device=device))
#                 continue
                
#             # Assign targets using TaskAlignedAssigner
#             assigned = self.assigner(
#                 anchors,
#                 gt_boxes,
#                 gt_labels,
#                 imgs[0].new_ones(len(anchors)),  # anchor mask
#                 self.nc
#             )
            
#             target_labels.append(assigned[0])
#             target_bboxes.append(assigned[1])
#             target_masks.append(assigned[2])
            
#         return target_labels, target_bboxes, target_masks
        
#     @torch.no_grad()
#     def get_predictions(self, cls_outputs, reg_outputs, conf_thresh=0.25):
#         """Convert network outputs to detections."""
#         predictions = []
        
#         for cls_out, reg_out in zip(cls_outputs, reg_outputs):
#             # Apply sigmoid to classification scores
#             scores = torch.sigmoid(cls_out)
            
#             # Get max scores and corresponding classes
#             max_scores, pred_classes = scores.max(dim=-1)
            
#             # Filter by confidence threshold
#             mask = max_scores > conf_thresh
#             if not mask.any():
#                 continue
                
#             # Get filtered predictions
#             filtered_boxes = reg_out[mask]
#             filtered_scores = max_scores[mask]
#             filtered_classes = pred_classes[mask]
            
#             predictions.append({
#                 'boxes': filtered_boxes,
#                 'scores': filtered_scores,
#                 'labels': filtered_classes
#             })
            
#         return predictions

def build_targets(pred_cls: List[torch.Tensor], 
                 pred_box: List[torch.Tensor], 
                 targets: torch.Tensor,
                 strides: List[int]) -> Tuple[List[torch.Tensor]]:
    """Build training targets for each feature level."""
    # Implementation for target assignment would go here
    # This would handle dynamic target assignment based on 
    # box IoU and classification confidence
    pass

class QOBBHead(nn.Module):
    """Quaternion-aware Oriented Bounding Box Head for multiple feature levels."""
    def __init__(self, nc, ch, reg_max=16):
        super().__init__()
        self.nc = nc
        self.ch = ch
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max + 4  # Classes, bbox distributions, quaternions

        # Ensure hidden_dim is a multiple of 4
        self.hidden_dim = 256  # Must be a multiple of 4

        # Create OBB head for each feature level
        self.detect_layers = nn.ModuleList()
        for channels in ch:
            assert channels % 4 == 0, f"Input channels must be multiple of 4, got {channels}"
            head = nn.Sequential(
                # First conv block
                QConv2D(channels, self.hidden_dim, kernel_size=3, stride=1, padding=1),
                QBN(self.hidden_dim),
                QReLU(),

                # Second conv block
                QConv2D(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
                QBN(self.hidden_dim),
                QReLU(),

                # Final conv to get outputs
                QConv2D(self.hidden_dim, self.no, kernel_size=1, stride=1)
            )
            self.detect_layers.append(head)

    def forward(self, features):
        """
        Forward pass through the OBB head.

        Args:
            features (List[torch.Tensor]): List of input feature maps [P3, P4, P5]
                Each with shape (B, C, H, W) or (C, H, W)

        Returns:
            List[torch.Tensor]: List of OBB outputs for each feature level,
                each with shape (B, no, 4, H, W)
        """
        outputs = []
        for feature, layer in zip(features, self.detect_layers):
            # Handle different input shapes
            if feature.dim() == 3:
                # Assume shape [C, H, W], add batch dimension
                feature = feature.unsqueeze(0)  # [1, C, H, W]
            if feature.dim() == 4:
                B, C, H, W = feature.shape
                assert C % 4 == 0, f"Channel dimension must be multiple of 4, got {C}"
                C_quat = C // 4
                feature = feature.view(B, C_quat, 4, H, W).contiguous()  # [B, C_quat, 4, H, W]
            elif feature.dim() == 5:
                B, C, Q, H, W = feature.shape
                assert Q == 4, f"Expected quaternion dimension to be 4, got {Q}"
                assert C % 4 == 0, f"Channel dimension must be multiple of 4, got {C}"
            else:
                raise ValueError(f"Unexpected feature dimensions: {feature.dim()}D")

            # Reshape to [B*Q, C_quat, H, W]
            if feature.dim() == 5:
                B, C_quat, Q, H, W = feature.shape
                feature_reshaped = feature.permute(0, 2, 1, 3, 4).contiguous().view(B * Q, C_quat, H, W)  # [B*4, C_quat, H, W]
            else:
                raise ValueError(f"Unexpected feature dimensions after processing: {feature.dim()}D")

            # Process through OBB head
            out = layer(feature_reshaped)  # [B*4, no, H, W]

            # Reshape back to [B, no, 4, H, W]
            out = out.view(B, Q, self.no, H, W).permute(0, 2, 1, 3, 4)  # [B, no, 4, H, W]
            outputs.append(out)

        return outputs


