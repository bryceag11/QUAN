import torch
import torch.nn as nn
from loss.box_loss import BboxLoss, DetectionLoss
from utils.ops import xywh2xyxy, xyxy2xywh, dist2bbox, bbox2dist, make_anchors
from utils.metrics import bbox_iou
import numpy as np
import torch.nn.functional as F

def create_dummy_data(batch_size=2, h=20, w=20, num_classes=80, device='cuda'):
    """Create dummy data for testing loss functions."""
    # Calculate channels
    reg_max = 16
    num_anchors = h * w
    
    # Create standard predictions
    pred_dist = torch.randn(batch_size, num_anchors, 4 * reg_max, device=device)
    pred_scores = torch.randn(batch_size, num_anchors, num_classes, device=device)
    
    # Create quaternion predictions
    pred_quat = torch.randn(batch_size, num_anchors, 4, device=device)
    pred_quat = F.normalize(pred_quat, p=2, dim=-1)  # Normalize quaternions
    
    # Combine predictions
    predictions = {
        'pred_dist': pred_dist,
        'pred_scores': pred_scores,
        'pred_quat': pred_quat
    }
    
    # Create target boxes and classes
    target_boxes = torch.tensor([
        [[0.2, 0.2, 0.4, 0.4],    # First image, first box
         [0.6, 0.6, 0.8, 0.8]],   # First image, second box
        [[0.3, 0.3, 0.5, 0.5],    # Second image, first box
         [0.7, 0.7, 0.9, 0.9]]    # Second image, second box
    ], device=device)
    
    target_cls = torch.tensor([
        [0, 1],  # First image classes
        [2, 3]   # Second image classes
    ], device=device)
    
    # Create target quaternions
    target_quats = torch.randn(batch_size, 2, 4, device=device)  # 2 boxes per image
    target_quats = F.normalize(target_quats, p=2, dim=-1)
    
    # Combine targets
    targets = {
        'boxes': target_boxes,
        'classes': target_cls,
        'quats': target_quats
    }
    
    return predictions, targets

def test_bbox_conversions():
    """Test bbox format conversions."""
    print("\n=== Testing bbox format conversions ===")
    
    boxes_xywh = torch.tensor([[100, 100, 50, 30],  
                              [200, 200, 40, 60]], dtype=torch.float32)
    print("Original boxes (xywh):", boxes_xywh)
    
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    print("Converted to xyxy:", boxes_xyxy)
    
    boxes_xywh_back = xyxy2xywh(boxes_xyxy)
    print("Converted back to xywh:", boxes_xywh_back)
    
    conversion_error = torch.abs(boxes_xywh - boxes_xywh_back).sum()
    print(f"Conversion error: {conversion_error:.6f}")

def test_bbox_iou():
    """Test IoU calculations."""
    print("\n=== Testing bbox IoU calculations ===")
    
    box1 = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    box2 = torch.tensor([[150, 150, 250, 250]], dtype=torch.float32)
    
    iou = bbox_iou(box1, box2, xywh=False)
    print("Overlapping boxes IoU:", iou)
    
    box3 = torch.tensor([[300, 300, 400, 400]], dtype=torch.float32)
    iou_no_overlap = bbox_iou(box1, box3, xywh=False)
    print("Non-overlapping boxes IoU:", iou_no_overlap)

def test_bbox_loss():
    """Test BboxLoss with detailed output."""
    print("\n=== Testing BboxLoss ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize loss functions
    bbox_loss_standard = BboxLoss(reg_max=16, nc=80, use_quat=False).to(device)
    bbox_loss_quat = BboxLoss(reg_max=16, nc=80, use_quat=True).to(device)
    print("Initialized BboxLoss modules")
    
    # Create dummy data with consistent shapes
    batch_size = 2
    h = w = 20
    num_anchors = h * w
    reg_max = 16
    
    # Create predictions
    pred_dist = torch.randn(batch_size, num_anchors, 4 * reg_max, device=device)
    pred_dist_reshaped = pred_dist.view(batch_size, num_anchors, 4, reg_max).softmax(3)
    pred_dist_decoded = (pred_dist_reshaped * torch.arange(reg_max, device=device)).sum(3)
    
    # Create anchor points
    anchor_points = torch.rand(num_anchors, 2, device=device)
    
    # Create target scores and masks
    target_scores = torch.rand(batch_size, num_anchors, device=device)
    fg_mask = torch.rand(batch_size, num_anchors) > 0.5
    fg_mask = fg_mask.to(device)
    
    # Ensure at least one positive sample
    fg_mask[0, 0] = True
    
    # Create target boxes (2 boxes per image)
    target_boxes = torch.rand(batch_size, num_anchors, 4, device=device)
    
    try:
        print("\nShape information:")
        print(f"pred_dist shape: {pred_dist.shape}")
        print(f"pred_dist_decoded shape: {pred_dist_decoded.shape}")
        print(f"anchor_points shape: {anchor_points.shape}")
        print(f"fg_mask shape: {fg_mask.shape}")
        print(f"fg_mask sum: {fg_mask.sum()}")
        
        print("\nTesting standard BboxLoss...")
        # Decode boxes with proper shapes
        pred_bboxes = dist2bbox(pred_dist_decoded, anchor_points)
        print(f"pred_bboxes shape: {pred_bboxes.shape}")
        
        loss_components = bbox_loss_standard(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_boxes,
            target_scores,
            target_scores.sum(),
            fg_mask
        )
        
        print("Standard BboxLoss components:")
        print(f"Box loss: {loss_components[0]:.4f}")
        print(f"DFL loss: {loss_components[1]:.4f}")
        
        print("\nTesting quaternion BboxLoss...")
        pred_quat = torch.randn(batch_size, num_anchors, 4, device=device)
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        target_quats = torch.randn(batch_size, num_anchors, 4, device=device)
        target_quats = F.normalize(target_quats, p=2, dim=-1)
        
        loss_components_quat = bbox_loss_quat(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_boxes,
            target_scores,
            target_scores.sum(),
            fg_mask,
            pred_quat,
            target_quats
        )
        
        print("Quaternion BboxLoss components:")
        print(f"Box loss: {loss_components_quat[0]:.4f}")
        print(f"DFL loss: {loss_components_quat[1]:.4f}")
        print(f"Quaternion loss: {loss_components_quat[2]:.4f}")
        
    except Exception as e:
        print(f"\nError in BboxLoss test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e



def test_bbox_loss():
    """Test BboxLoss with detailed output."""
    print("\n=== Testing BboxLoss ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize loss functions
    bbox_loss_standard = BboxLoss(reg_max=16, nc=80, use_quat=False).to(device)
    bbox_loss_quat = BboxLoss(reg_max=16, nc=80, use_quat=True).to(device)
    print("Initialized BboxLoss modules")
    
    # Create dummy data
    batch_size = 2
    h = w = 20
    num_anchors = h * w  # 400
    reg_max = 16
    
    # Create predictions with proper shapes
    pred_dist = torch.randn(batch_size, num_anchors, 4 * reg_max, device=device)
    
    anchor_points = torch.rand(num_anchors, 2, device=device)
    
    # Initialize fg_mask to mark only the assigned anchors as positive
    fg_mask = torch.zeros(batch_size, num_anchors, device=device, dtype=torch.bool)
    fg_mask[0, 0] = True
    fg_mask[0, 1] = True
    fg_mask[1, 0] = True
    fg_mask[1, 1] = True
    # Now fg_mask.sum()=4
    
    # Set target_scores: 1.0 for positive anchors, 0.0 otherwise
    target_scores = torch.zeros(batch_size, num_anchors, device=device)
    target_scores[fg_mask] = 1.0  # Only positive anchors have score=1.0
    
    # Create target boxes with shape [batch_size, num_anchors, 4]
    target_boxes = torch.zeros(batch_size, num_anchors, 4, device=device)
    
    # Assign the two target boxes to the first two anchors of each image
    target_boxes[0, 0, :] = torch.tensor([0.2, 0.2, 0.4, 0.4], device=device)
    target_boxes[0, 1, :] = torch.tensor([0.6, 0.6, 0.8, 0.8], device=device)
    target_boxes[1, 0, :] = torch.tensor([0.3, 0.3, 0.5, 0.5], device=device)
    target_boxes[1, 1, :] = torch.tensor([0.7, 0.7, 0.9, 0.9], device=device)
    
    try:
        print("\nShape information:")
        print(f"pred_dist shape: {pred_dist.shape}")  # [2, 400, 64]
        print(f"anchor_points shape: {anchor_points.shape}")  # [400, 2]
        print(f"fg_mask shape: {fg_mask.shape}")  # [2, 400]
        print(f"fg_mask sum: {fg_mask.sum()}")  # 4
        print(f"target_boxes shape: {target_boxes.shape}")  # [2, 400, 4]
        print(f"target_boxes non-zero count: {(target_boxes != 0).sum()}")  # 16 (4 boxes * 4 values)
        
        print("\nTesting standard BboxLoss...")
        # Decode boxes with proper shapes by passing raw pred_dist
        pred_bboxes = bbox_loss_standard.bbox_decode(pred_dist, anchor_points)
        print(f"pred_bboxes shape: {pred_bboxes.shape}")  # [2, 400, 4]
        
        loss_components = bbox_loss_standard(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_boxes,
            target_scores,
            target_scores.sum(),
            fg_mask
        )
        
        print("Standard BboxLoss components:")
        print(f"Box loss: {loss_components[0]:.4f}")
        print(f"DFL loss: {loss_components[1]:.4f}")
        
        print("\nTesting quaternion BboxLoss...")
        # Add quaternion predictions
        pred_quat = torch.randn(batch_size, num_anchors, 4, device=device)
        pred_quat = F.normalize(pred_quat, p=2, dim=-1)
        target_quats = torch.randn(batch_size, num_anchors, 4, device=device)
        target_quats = F.normalize(target_quats, p=2, dim=-1)
        
        loss_components_quat = bbox_loss_quat(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_boxes,
            target_scores,
            target_scores.sum(),
            fg_mask,
            pred_quat,
            target_quats
        )
        
        print("Quaternion BboxLoss components:")
        print(f"Box loss: {loss_components_quat[0]:.4f}")
        print(f"DFL loss: {loss_components_quat[1]:.4f}")
        print(f"Quaternion loss: {loss_components_quat[2]:.4f}")
        
    except Exception as e:
        print(f"\nError in BboxLoss test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


def test_detection_loss():
    """Test DetectionLoss with detailed output."""
    print("\n=== Testing DetectionLoss ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize DetectionLoss directly with parameters
    detection_loss = DetectionLoss(reg_max=16, nc=80, use_quat=True, tal_topk=10).to(device)
    print("Initialized DetectionLoss module")
    
    # Create dummy data
    batch_size = 2
    num_anchors = 400  # e.g., 20x20 grid
    reg_max = 16
    nc = 80
    
    # Create dummy predictions
    # pred_dist: [batch_size, num_anchors, 4 * reg_max]
    pred_dist = torch.randn(batch_size, num_anchors, 4 * reg_max, device=device)
    
    # Create dummy decoded bounding boxes
    # For testing purposes, let's assume bbox_decode returns random boxes
    pred_bboxes = torch.randn(batch_size, num_anchors, 4, device=device)
    
    # Create dummy anchor points
    # anchor_points: [num_anchors, 2]
    anchor_points = torch.rand(num_anchors, 2, device=device)
    
    # Create dummy target bounding boxes
    # target_bboxes: [batch_size, num_anchors, 4]
    target_bboxes = torch.zeros(batch_size, num_anchors, 4, device=device)
    # Assign some boxes to specific anchors
    target_bboxes[0, 0, :] = torch.tensor([0.2, 0.2, 0.4, 0.4], device=device)
    target_bboxes[0, 1, :] = torch.tensor([0.6, 0.6, 0.8, 0.8], device=device)
    target_bboxes[1, 0, :] = torch.tensor([0.3, 0.3, 0.5, 0.5], device=device)
    target_bboxes[1, 1, :] = torch.tensor([0.7, 0.7, 0.9, 0.9], device=device)
    
    # Create dummy objectness scores
    # target_scores: [batch_size, num_anchors]
    target_scores = torch.zeros(batch_size, num_anchors, device=device)
    target_scores[0, 0] = 1.0
    target_scores[0, 1] = 1.0
    target_scores[1, 0] = 1.0
    target_scores[1, 1] = 1.0
    target_scores_sum = target_scores.sum()
    
    # Create dummy foreground mask
    # fg_mask: [batch_size, num_anchors]
    fg_mask = torch.zeros(batch_size, num_anchors, device=device, dtype=torch.bool)
    fg_mask[0, 0] = True
    fg_mask[0, 1] = True
    fg_mask[1, 0] = True
    fg_mask[1, 1] = True
    
    # Create dummy quaternions
    # pred_quat and target_quat: [batch_size, num_anchors, 4]
    pred_quat = F.normalize(torch.randn(batch_size, num_anchors, 4, device=device), p=2, dim=-1)
    target_quat = F.normalize(torch.randn(batch_size, num_anchors, 4, device=device), p=2, dim=-1)
    
    try:
        print("\nShape information:")
        print(f"pred_dist shape: {pred_dist.shape}")  # [2, 400, 64]
        print(f"pred_bboxes shape: {pred_bboxes.shape}")  # [2, 400, 4]
        print(f"anchor_points shape: {anchor_points.shape}")  # [400, 2]
        print(f"fg_mask shape: {fg_mask.shape}")  # [2, 400]
        print(f"fg_mask sum: {fg_mask.sum()}")  # 4
        print(f"target_bboxes shape: {target_bboxes.shape}")  # [2, 400, 4]
        print(f"target_bboxes non-zero count: {(target_bboxes != 0).sum()}")  # 16
        print(f"target_scores shape: {target_scores.shape}")  # [2, 400]
        print(f"target_scores_sum: {target_scores_sum.item()}")  # 4.0
        print(f"pred_quat shape: {pred_quat.shape}")  # [2, 400, 4]
        print(f"target_quat shape: {target_quat.shape}")  # [2, 400, 4]
        
        print("\nTesting DetectionLoss...")
        # Compute loss
        box_loss, dfl_loss, quat_loss = detection_loss(
            pred_dist, 
            pred_bboxes, 
            anchor_points, 
            target_bboxes, 
            target_scores, 
            target_scores_sum, 
            fg_mask, 
            pred_quat, 
            target_quat
        )
        
        print("DetectionLoss components:")
        print(f"Box loss: {box_loss.item():.4f}")
        print(f"DFL loss: {dfl_loss.item():.4f}")
        if detection_loss.use_quat:
            print(f"Quaternion loss: {quat_loss.item():.4f}")
    except Exception as e:
        print(f"\nError in DetectionLoss test: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


def test_full_training_loop():
    """Test losses in a mock training loop."""
    print("\n=== Testing Losses in Training Loop ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detection_loss = DetectionLoss(nc=80, reg_max=16, use_quat=True).to(device)
    
    # Mock training loop
    print("\nRunning mock training loop...")
    for i in range(3):
        predictions, targets = create_dummy_data(device=device)
        
        # Calculate loss
        with torch.cuda.amp.autocast():
            loss = detection_loss(predictions, targets)
        
        total_loss = loss.sum()
        print(f"\nIteration {i+1}:")
        print(f"Total loss: {total_loss:.4f}")
        print(f"Box loss: {loss[0]:.4f}")
        print(f"Classification loss: {loss[1]:.4f}")
        print(f"DFL loss: {loss[2]:.4f}")
        print(f"Quaternion loss: {loss[3]:.4f}")

def test_dist2bbox():
    """Test distance to bbox conversion."""
    print("\n=== Testing dist2bbox conversion ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create sample inputs
    anchor_points = torch.tensor([[10., 10.], [20., 20.]], device=device)
    distance = torch.tensor([[2., 2., 1., 1.], [3., 3., 1.5, 1.5]], device=device)
    
    try:
        # Test single batch
        boxes = dist2bbox(distance, anchor_points, xywh=True)
        print(f"Single batch output shape: {boxes.shape}")
        print(f"Sample boxes:\n{boxes}")
        
        # Test batched input
        distance_batch = distance.unsqueeze(0).expand(2, -1, -1)
        boxes_batch = dist2bbox(distance_batch, anchor_points, xywh=True)
        print(f"\nBatched output shape: {boxes_batch.shape}")
        
    except Exception as e:
        print(f"Error in dist2bbox test: {str(e)}")
        raise e


def main():
    """Run all tests."""
    print("Starting loss function tests...")
    
    try:
        # test_dist2bbox()
        # test_bbox_conversions()
        # test_bbox_iou()
        # test_anchor_generation()
        test_detection_loss()
        test_bbox_loss()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        raise e

if __name__ == "__main__":
    main()