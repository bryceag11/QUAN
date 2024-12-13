# models/heads/qdet_head.py

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from utils.ops import dist2bbox, dist2rbox, make_anchors
from quaternion.conv import QConv2D  # Import Quaternion-aware Conv layer
from quaternion.qactivation import QReLU  # Import Quaternion-aware activation
from quaternion.qbatch_norm import QBN  # Import Quaternion-aware BatchNorm
from typing import List, Union
from quaternion.qbatch_norm import IQBN

class QDetectHead(nn.Module):
    """Quaternion-aware Detection Head for multiple feature levels."""
    def __init__(self, nc, ch, reg_max=16):
        super().__init__()
        self.nc = nc  # number of classes
        self.reg_max = reg_max
        self.no = nc + 4 * reg_max  # outputs per anchor
        self.nl = len(ch)  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        # Create detection sub-networks
        self.detect_layers = nn.ModuleList()
        for channels in ch:
            assert channels % 4 == 0, f"Channel count must be multiple of 4, got {channels}"
            head = nn.Sequential(
                QConv2D(channels, channels, 3, stride=1, padding=1),
                IQBN(channels // 4),
                QReLU(),
                QConv2D(channels, channels, 3, stride=1, padding=1),
                IQBN(channels // 4),
                QReLU(),
                QConv2D(channels, self.no, 1)  # Final layer for predictions
            )
            self.detect_layers.append(head)
            
        # Initialize anchors
        self.anchors = torch.empty(0)
        self.anchor_points = {}  # Dictionary to store anchor points for each layer
        self.strides = torch.tensor([8, 16, 32])  # Standard strides for P3, P4, P5

    def forward(self, features):
        """
        Forward pass of the detection head.
        
        Args:
            features: List of feature maps [P3, P4, P5]
                Each with shape (B, C, 4, H, W)
        
        Returns:
            List of detection outputs for each feature level
        """
        outputs = []
        # Process each feature level
        for i, (feat, layer, stride) in enumerate(zip(features, self.detect_layers, self.strides)):
            # Generate anchor points if not already computed
            if i not in self.anchor_points:
                h, w = feat.shape[-2:]
                self.anchor_points[i] = self._make_anchors(h, w, stride, feat.device)
            
            # Get predictions
            out = layer(feat)  # [B, no, 4, H, W]
            outputs.append(out)
        
        return outputs, self.anchor_points

    def _make_anchors(self, h, w, stride, device):
        """Generate anchor points for a given feature map."""
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        )
        
        grid_xy = torch.stack([
            (grid_x + 0.5) * stride,  # Shift by 0.5 and scale by stride
            (grid_y + 0.5) * stride
        ], dim=-1).float()
        
        return grid_xy.reshape(-1, 2)  # [H*W, 2]


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


# class Classify(nn.Module):
#     """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
#         """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
#         super().__init__()
#         c_ = 1280  # EfficientNet-B0 size, ensure it's divisible by 4
#         self.conv = QConv2D(c1, c_, kernel_size=k, stride=s, padding=p, groups=g)
#         self.bn = QBatchNorm2d(c_)
#         self.act = QReLU()
#         self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
#         self.drop = nn.Dropout(p=0.0, inplace=True)
#         self.linear = nn.Linear(c_, c2)  # to x(b,c2)

#     def forward(self, x):
#         """Performs a forward pass of the YOLO classification head on input quaternion data."""
#         if isinstance(x, list):
#             x = torch.cat(x, dim=1)  # Concatenate along channel dimension
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.act(x)
#         x = self.pool(x).flatten(1)  # Shape: (B, c_)
#         x = self.drop(x)
#         x = self.linear(x)  # Shape: (B, c2)
#         return x if self.training else x.softmax(dim=1)

# class Detect(nn.Module):
#     """YOLO Detect head for detection models."""

#     dynamic = False  # force grid reconstruction
#     export = False  # export mode
#     end2end = False  # end2end
#     max_det = 300  # max_det
#     shape = None
#     anchors = torch.empty(0)  # init
#     strides = torch.empty(0)  # init
#     legacy = False  # backward compatibility for v3/v5/v8/v9 models

#     def __init__(self, nc=80, ch=()):
#         """Initializes the YOLO detection layer with specified number of classes and channels."""
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(ch)  # number of detection layers
#         self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
#         self.no = nc + self.reg_max * 4  # number of outputs per anchor
#         self.stride = torch.zeros(self.nl)  # strides computed during build
#         c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
#         self.cv2 = nn.ModuleList(
#             nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
#         )
#         self.cv3 = (
#             nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
#             if self.legacy
#             else nn.ModuleList(
#                 nn.Sequential(
#                     nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
#                     nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
#                     nn.Conv2d(c3, self.nc, 1),
#                 )
#                 for x in ch
#             )
#         )
#         self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#         if self.end2end:
#             self.one2one_cv2 = copy.deepcopy(self.cv2)
#             self.one2one_cv3 = copy.deepcopy(self.cv3)

#     def forward(self, x):
#         """Concatenates and returns predicted bounding boxes and class probabilities."""
#         if self.end2end:
#             return self.forward_end2end(x)

#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return x
#         y = self._inference(x)
#         return y if self.export else (y, x)

#     def forward_end2end(self, x):
#         """
#         Performs forward pass of the v10Detect module.

#         Args:
#             x (tensor): Input tensor.

#         Returns:
#             (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
#                            If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
#         """
#         x_detach = [xi.detach() for xi in x]
#         one2one = [
#             torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
#         ]
#         for i in range(self.nl):
#             x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
#         if self.training:  # Training path
#             return {"one2many": x, "one2one": one2one}

#         y = self._inference(one2one)
#         y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
#         return y if self.export else (y, {"one2many": x, "one2one": one2one})

#     def _inference(self, x):
#         """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
#         # Inference path
#         shape = x[0].shape  # BCHW
#         x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
#         if self.dynamic or self.shape != shape:
#             self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
#             self.shape = shape

#         if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
#             box = x_cat[:, : self.reg_max * 4]
#             cls = x_cat[:, self.reg_max * 4 :]
#         else:
#             box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

#         if self.export and self.format in {"tflite", "edgetpu"}:
#             # Precompute normalization factor to increase numerical stability
#             # See https://github.com/ultralytics/ultralytics/issues/7371
#             grid_h = shape[2]
#             grid_w = shape[3]
#             grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
#             norm = self.strides / (self.stride[0] * grid_size)
#             dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
#         else:
#             dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

#         return torch.cat((dbox, cls.sigmoid()), 1)

#     def bias_init(self):
#         """Initialize Detect() biases, WARNING: requires stride availability."""
#         m = self  # self.model[-1]  # Detect() module
#         # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
#         # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
#         for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
#             a[-1].bias.data[:] = 1.0  # box
#             b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
#         if self.end2end:
#             for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
#                 a[-1].bias.data[:] = 1.0  # box
#                 b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

#     def decode_bboxes(self, bboxes, anchors):
#         """Decode bounding boxes."""
#         return dist2bbox(bboxes, anchors, xywh=not self.end2end, dim=1)

#     @staticmethod
#     def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
#         """
#         Post-processes YOLO model predictions.

#         Args:
#             preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
#                 format [x, y, w, h, class_probs].
#             max_det (int): Maximum detections per image.
#             nc (int, optional): Number of classes. Default: 80.

#         Returns:
#             (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
#                 dimension format [x, y, w, h, max_class_prob, class_index].
#         """
#         batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
#         boxes, scores = preds.split([4, nc], dim=-1)
#         index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
#         boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
#         scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
#         scores, index = scores.flatten(1).topk(min(max_det, anchors))
#         i = torch.arange(batch_size)[..., None]  # batch indices
#         return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)

# class v10Detect(Detect):
#     """
#     v10 Detection head from https://arxiv.org/pdf/2405.14458.

#     Args:
#         nc (int): Number of classes.
#         ch (tuple): Tuple of channel sizes.

#     Attributes:
#         max_det (int): Maximum number of detections.

#     Methods:
#         __init__(self, nc=80, ch=()): Initializes the v10Detect object.
#         forward(self, x): Performs forward pass of the v10Detect module.
#         bias_init(self): Initializes biases of the Detect module.

#     """

#     end2end = True

#     def __init__(self, nc=80, ch=()):
#         """Initializes the v10Detect object with the specified number of classes and input channels."""
#         super().__init__(nc, ch)
#         c3 = max(ch[0], min(self.nc, 100))  # channels
#         # Light cls head
#         self.cv3 = nn.ModuleList(
#             nn.Sequential(
#                 nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
#                 nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
#                 nn.Conv2d(c3, self.nc, 1),
#             )
#             for x in ch
#         )
#         self.one2one_cv3 = copy.deepcopy(self.cv3)
