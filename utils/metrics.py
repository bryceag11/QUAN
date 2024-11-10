# utils/metrics.py

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn
from utils import LOGGER, SimpleClass, TryExcept, plt_settings
from .ops import xywh2xyxy
# Removed OKS_SIGMA and kpt_iou as they are not relevant for OBB-based detection
import shapely.geometry

def quaternion_to_angle(quat):
    """
    Convert quaternions to rotation angles in radians.

    Args:
        quat (torch.Tensor): Quaternions, shape (N, 4), format [qx, qy, qz, qw].

    Returns:
        torch.Tensor: Rotation angles in radians, shape (N,).
    """
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    theta = 2 * torch.atan2(qz, qw)  # Assuming rotation around z-axis
    return theta

def obb_to_polygon(box):
    """
    Convert an Oriented Bounding Box (OBB) to a Shapely Polygon.

    Args:
        box (numpy.ndarray): An array of shape (8,) representing a single OBB.
                             Format: [x, y, w, h, qx, qy, qz, qw]

    Returns:
        shapely.geometry.Polygon: The polygon representation of the OBB.
    """
    x, y, w, h, qx, qy, qz, qw = box

    # Convert quaternion to rotation angle
    theta = 2 * math.atan2(qz, qw)

    # Compute rotation matrix components
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Define the four corners of the box before rotation
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])

    # Rotate corners
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_corners = np.dot(corners, rotation_matrix.T)

    # Translate to (x, y)
    translated_corners = rotated_corners + np.array([x, y])

    return shapely.geometry.Polygon(translated_corners)



def quaternion_bbox_ioa(box1, box2, quats1, quats2, iou=False, eps=1e-7):
    """
    Calculate the intersection over box2 area given OBBs with quaternions.

    Args:
        box1 (torch.Tensor): Bounding boxes, shape (N, 8), format [x, y, w, h, qx, qy, qz, qw].
        box2 (torch.Tensor): Bounding boxes, shape (M, 8), format [x, y, w, h, qx, qy, qz, qw].
        quats1 (torch.Tensor): Quaternions for box1, shape (N, 4).
        quats2 (torch.Tensor): Quaternions for box2, shape (M, 4).
        iou (bool): Calculate standard IoU if True, else inter_area/box2_area.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: Intersection over box2 area, shape (N, M).
    """
    if not isinstance(box1, torch.Tensor):
        box1 = torch.from_numpy(box1).float()
    if not isinstance(box2, torch.Tensor):
        box2 = torch.from_numpy(box2).float()

    N = box1.shape[0]
    M = box2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), device=box1.device)

    # Move to CPU for shapely operations
    box1_np = box1.cpu().numpy()
    box2_np = box2.cpu().numpy()

    iou_matrix = torch.zeros((N, M), device=box1.device)

    for i in range(N):
        box1_poly = obb_to_polygon(box1_np[i])
        for j in range(M):
            box2_poly = obb_to_polygon(box2_np[j])
            inter = box1_poly.intersection(box2_poly).area
            box2_area = box2[j, 2] * box2[j, 3]  # width * height
            iou_matrix[i, j] = inter / (box2_area + eps)

    if iou:
        # If standard IoU is requested, compute the IoU using existing functions
        return box_iou(box1[:, :4], box2[:, :4], quats1=quats1, quats2=quats2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=eps)
    else:
        return iou_matrix


def bbox_ioa(box1, box2, iou=False, eps=1e-7, quats1=None, quats2=None):
    """
    Calculate the intersection over box2 area given box1 and box2. Boxes can be axis-aligned or OBBs with quaternions.
    """
    if quats1 is not None and quats2 is not None:
        # Implement OBB IoA with quaternions
        return quaternion_bbox_ioa(box1, box2, quats1, quats2, iou=iou, eps=eps)
    else:
        # Existing axis-aligned IoA computation
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

        inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * (
            np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)
        ).clip(0)

        area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        if iou:
            box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
            area = area + box1_area[:, None] - inter_area

        return inter_area / (area + eps)

def box_iou(box1, box2, quats1=None, quats2=None, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of boxes with optional quaternion support.
    """
    if quats1 is not None and quats2 is not None:
        # Implement quaternion-aware IoU
        return probiou(box1, box2, CIoU=CIoU, eps=eps)  # Ensure probiou handles quaternions
    else:
        # Existing axis-aligned IoU computation
        if xywh:
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp_(0) * (
            torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
        ).clamp_(0)

        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        if CIoU or DIoU or GIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            if CIoU or DIoU:
                c2 = cw.pow(2) + ch.pow(2) + eps
                rho2 = (
                    (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
                ) / 4
                if CIoU:
                    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)
                return iou - rho2 / c2
            c_area = cw * ch + eps
            return iou - (c_area - union) / c_area
        return iou

def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.
    """
    intersection = torch.matmul(mask1, mask2.T).clamp_(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection
    return intersection / (union + eps)

def _get_covariance_matrix(boxes):
    """
    Generate covariance matrices from Oriented Bounding Boxes (OBBs) with quaternions.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 8) representing rotated bounding boxes.
                              Format: [x, y, w, h, qx, qy, qz, qw]

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - a (torch.Tensor): Shape (N,), represents a*cos^2(theta) + b*sin^2(theta)
            - b (torch.Tensor): Shape (N,), represents a*sin^2(theta) + b*cos^2(theta)
            - c (torch.Tensor): Shape (N,), represents (a - b)*sin(theta)*cos(theta)
    """
    # Extract width and height
    w = boxes[:, 2]
    h = boxes[:, 3]

    # Extract quaternions
    quats = boxes[:, 4:8]

    # Convert quaternions to rotation angles
    theta = quaternion_to_angle(quats)  # Shape: (N,)

    # Compute a and b
    a = (w ** 2) / 12.0  # Shape: (N,)
    b = (h ** 2) / 12.0  # Shape: (N,)

    # Compute cosine and sine of rotation angles
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Compute covariance matrix components
    a_cos2 = a * (cos_theta ** 2)  # a*cos^2(theta)
    b_sin2 = b * (sin_theta ** 2)  # b*sin^2(theta)
    a_sin2 = a * (sin_theta ** 2)  # a*sin^2(theta)
    b_cos2 = b * (cos_theta ** 2)  # b*cos^2(theta)
    ab_cos_sin = (a - b) * sin_theta * cos_theta  # (a - b)*sin(theta)*cos(theta)

    return a_cos2, a_sin2 + b_cos2, ab_cos_sin

def probiou(obb1, obb2, CIoU=False, eps=1e-7):
    """
    Calculate probabilistic IoU between oriented bounding boxes.

    Implements the algorithm from https://arxiv.org/pdf/2106.06072v1.pdf.

    Args:
        obb1 (torch.Tensor): Ground truth OBBs, shape (N, 8), format [x, y, w, h, qx, qy, qz, qw].
        obb2 (torch.Tensor): Predicted OBBs, shape (N, 8), format [x, y, w, h, qx, qy, qz, qw].
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): Small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: OBB similarities, shape (N,).
    """
    # Compute covariance matrices
    a1, b1, c1 = _get_covariance_matrix(obb1)  # Shape: (N,)
    a2, b2, c2 = _get_covariance_matrix(obb2)  # Shape: (N,)

    # Extract center coordinates
    x1, y1 = obb1[:, 0], obb1[:, 1]
    x2, y2 = obb2[:, 0], obb2[:, 1]

    # Compute terms t1, t2, t3
    numerator = (a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)
    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = (numerator / denominator) * 0.25
    t2 = ((c1 + c2) * (x2 - x1) * (y1 - y2)) / denominator * 0.5
    t3 = ((a1 + a2) * (b1 + b2) - (c1 + c2).pow(2)) / (
        4 * ((a1 * b1 - c1.pow(2)).clamp(min=eps) * (a2 * b2 - c2.pow(2)).clamp(min=eps)).sqrt() + eps
    ) + eps
    t3 = t3.log() * 0.5

    bd = (t1 + t2 + t3).clamp(min=eps, max=100.0)
    hd = (1.0 - (-bd).exp() + eps).sqrt()
    iou = 1 - hd

    if CIoU:
        # Compute aspect ratio similarity v and alpha
        w1, h1 = obb1[:, 2], obb1[:, 3]
        w2, h2 = obb2[:, 2], obb2[:, 3]
        v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - v * alpha

    return iou

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    Calculate the probabilistic IoU between batches of oriented bounding boxes.

    Args:
        obb1 (torch.Tensor | np.ndarray): Ground truth OBBs, shape (N, 8), format [x, y, w, h, qx, qy, qz, qw].
        obb2 (torch.Tensor | np.ndarray): Predicted OBBs, shape (M, 8), format [x, y, w, h, qx, qy, qz, qw].
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        torch.Tensor: A tensor of shape (N, M) representing OBB similarities.
    """
    # Convert to torch tensors if numpy arrays
    if isinstance(obb1, np.ndarray):
        obb1 = torch.from_numpy(obb1).float()
    if isinstance(obb2, np.ndarray):
        obb2 = torch.from_numpy(obb2).float()

    N = obb1.shape[0]
    M = obb2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), device=obb1.device)

    # Expand dimensions to compute pairwise probiou
    obb1_exp = obb1.unsqueeze(1).expand(N, M, -1)  # Shape: (N, M, 8)
    obb2_exp = obb2.unsqueeze(0).expand(N, M, -1)  # Shape: (N, M, 8)

    # Reshape to (N*M, 8)
    obb1_flat = obb1_exp.reshape(N * M, -1)
    obb2_flat = obb2_exp.reshape(N * M, -1)

    # Compute probiou for each pair
    iou_flat = probiou(obb1_flat, obb2_flat, CIoU=False, eps=eps)  # Shape: (N*M,)

    # Reshape back to (N, M)
    iou = iou_flat.reshape(N, M)

    return iou


def smooth(y, f=0.05):
    """Box filter of fraction f."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


@plt_settings()
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names={}, on_plot=None):
    """Plots a precision-recall curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


@plt_settings()
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names={}, xlabel="Confidence", ylabel="Metric", on_plot=None):
    """Plots a metric-confidence curve."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if on_plot:
        on_plot(save_dir)


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec



def ap_per_class(
    tp, conf, pred_cls, target_cls, plot=False, on_plot=None, save_dir=Path(), names={}, eps=1e-16, prefix=""
):
    """
    Computes the average precision per class for object detection evaluation, including OBB with quaternions.
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]
    x, prec_values = np.linspace(0, 1, 1000), []
    ap, p_curve, r_curve = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]
        n_p = i.sum()
        if n_p == 0 or n_l == 0:
            continue
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if j == 0:
                prec_values.append(np.interp(x, mrec, mpre))
    prec_values = np.array(prec_values)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]
    names = dict(enumerate(names))
    if plot:
        plot_pr_curve(x, prec_values, ap, save_dir / f"{prefix}PR_curve.png", names, on_plot=on_plot)
        plot_mc_curve(x, f1_curve, save_dir / f"{prefix}F1_curve.png", names, ylabel="F1", on_plot=on_plot)
        plot_mc_curve(x, p_curve, save_dir / f"{prefix}P_curve.png", names, ylabel="Precision", on_plot=on_plot)
        plot_mc_curve(x, r_curve, save_dir / f"{prefix}R_curve.png", names, ylabel="Recall", on_plot=on_plot)
    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x, prec_values


def smooth_BCE(eps=0.1):
    """
    Computes smoothed positive and negative Binary Cross-Entropy targets.

    This function calculates positive and negative label smoothing BCE targets based on a given epsilon value.
    For implementation details, refer to https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.

    Args:
        eps (float, optional): The epsilon value for label smoothing. Defaults to 0.1.

    Returns:
        (tuple): A tuple containing the positive and negative label smoothing BCE targets.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        self.iou_thres = iou_thres

    def process_cls_preds(self, preds, targets):
        """
        Update confusion matrix for classification task.

        Args:
            preds (Array[N, min(nc,5)]): Predicted class labels.
            targets (Array[N, 1]): Ground truth class labels.
        """
        preds, targets = torch.cat(preds)[:, 0], torch.cat(targets)
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[p][t] += 1

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # remove background class if task=detect

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))



class Metric(SimpleClass):
    """
    Class for computing evaluation metrics for YOLOv8 model.
    """
    def __init__(self) -> None:
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []
        self.nc = 0

    @property
    def ap50(self):
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        w = [0.0, 0.0, 0.1, 0.9]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    @property
    def curves(self):
        return []

    @property
    def curves_results(self):
        return [
            [self.px, self.prec_values, "Recall", "Precision"],
            [self.px, self.f1_curve, "Confidence", "F1"],
            [self.px, self.p_curve, "Confidence", "Precision"],
            [self.px, self.r_curve, "Confidence", "Recall"],
        ]



class DetMetrics(SimpleClass):
    """
    Utility class for computing detection metrics such as precision, recall, and mean average precision (mAP).
    """
    def __init__(self, save_dir=Path("."), plot=False, on_plot=None, names={}) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.on_plot = on_plot
        self.names = names
        self.box = Metric()
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.task = "detect"

    def process(self, tp, conf, pred_cls, target_cls):
        results = ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            plot=self.plot,
            save_dir=self.save_dir,
            names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        return ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]

    def mean_results(self):
        return self.box.mean_results()

    def class_result(self, i):
        return self.box.class_result(i)

    @property
    def maps(self):
        return self.box.maps

    @property
    def fitness(self):
        return self.box.fitness()

    @property
    def ap_class_index(self):
        return self.box.ap_class_index

    @property
    def results_dict(self):
        return dict(zip(self.keys + ["fitness"], self.mean_results() + [self.fitness]))

    @property
    def curves(self):
        return ["Precision-Recall(B)", "F1-Confidence(B)", "Precision-Confidence(B)", "Recall-Confidence(B)"]

    @property
    def curves_results(self):
        return self.box.curves_results

# utils/metrics.py

import torch

class OBBMetrics:
    def __init__(self, save_dir, plot=True, names=None):
        """
        Initialize the Metrics class.

        Args:
            save_dir (str): Directory to save metrics and plots.
            plot (bool, optional): Whether to generate plots.
            names (dict, optional): Mapping from class indices to class names.
        """
        self.save_dir = save_dir
        self.plot = plot
        self.names = names if names is not None else {}
        self.reset()
    
    def reset(self):
        """
        Reset all metrics.
        """
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.total = 0
        # Add other metric-related initializations as needed
    
    def update(self, preds, target_bboxes, target_categories, target_quats, bbox_types):
        """
        Update metrics with predictions and targets.

        Args:
            preds (torch.Tensor): Predictions from the model.
            target_bboxes (torch.Tensor): Ground truth bounding boxes.
            target_categories (torch.Tensor): Ground truth categories.
            target_quats (torch.Tensor): Ground truth quaternions.
            bbox_types (torch.Tensor): Type of bounding boxes (0: OBB, 1: XYWH).
        """
        # Implement metric updates (e.g., IoU calculations, precision, recall)
        # Placeholder implementation
        # You need to replace this with actual metric computations
        pass

    def compute(self):
        """
        Compute final metrics.
        """
        # Implement final metric computations (e.g., mAP)
        pass

    def visualize(self, save_dir):
        """
        Generate and save visualizations for metrics.

        Args:
            save_dir (str): Directory to save visualizations.
        """
        if self.plot:
            # Implement visualization (e.g., PR curves, confusion matrices)
            pass

    def plot_final_metrics(self, save_dir, names):
        """
        Plot and save final metrics.

        Args:
            save_dir (str): Directory to save plots.
            names (dict): Mapping from class indices to class names.
        """
        if self.plot:
            # Implement final metric plots
            pass



def bbox_iou(box1, box2, quats1=None, quats2=None, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of two sets of bounding boxes.

    Args:
        box1 (torch.Tensor): Bounding boxes, shape (N, 4).
        box2 (torch.Tensor): Bounding boxes, shape (M, 4).
        quats1 (torch.Tensor, optional): Quaternions for box1, shape (N, 4).
        quats2 (torch.Tensor, optional): Quaternions for box2, shape (M, 4).
        xywh (bool, optional): If True, boxes are in [x, y, w, h] format.
        GIoU, DIoU, CIoU (bool, optional): Additional IoU variants.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU matrix, shape (N, M).
    """
    if quats1 is not None and quats2 is not None:
        # Implement quaternion-aware IoU
        return quaternion_bbox_ioa(box1, box2, quats1, quats2, iou=False, eps=eps)
    else:
        # Use standard box IoU
        return standard_box_iou(box1, box2, xywh=xywh, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU, eps=eps)

def standard_box_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate standard IoU for axis-aligned bounding boxes.

    Args:
        box1 (torch.Tensor): Bounding boxes, shape (N, 4).
        box2 (torch.Tensor): Bounding boxes, shape (M, 4).
        xywh (bool, optional): If True, boxes are in [x, y, w, h] format.
        GIoU, DIoU, CIoU (bool, optional): Additional IoU variants.
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        torch.Tensor: IoU matrix, shape (N, M).
    """
    if xywh:
        # Convert [x, y, w, h] to [x1, y1, x2, y2]
        box1 = xywh2xyxy(box1)
        box2 = xywh2xyxy(box2)

    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=1)

    inter_x1 = torch.max(b1_x1, b2_x1.t())
    inter_y1 = torch.max(b1_y1, b2_y1.t())
    inter_x2 = torch.min(b1_x2, b2_x2.t())
    inter_y2 = torch.min(b1_y2, b2_y2.t())

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = area1 + area2.t() - inter_area + eps
    iou = inter_area / union_area

    # Handle GIoU, DIoU, CIoU if needed
    # Placeholder: Implement GIoU, DIoU, CIoU as per your requirements

    return iou
