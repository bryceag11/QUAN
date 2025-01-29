# test2.py

import torch
import torch.nn.functional as F
from loss.box_loss import DetectionLoss
from utils.ops import make_anchors, dist2bbox, bbox2dist
import matplotlib.pyplot as plt

def test_distance_conversion():
    """Test bbox to distance and back conversion"""
    print("\nTesting Distance Conversion...")
    
    # Create sample anchor points and boxes
    anchors = torch.tensor([
        [10., 10.],
        [20., 20.],
        [30., 30.]
    ])
    
    boxes = torch.tensor([
        [12., 12., 4., 4.],  # Small box near first anchor
        [22., 22., 6., 6.],  # Medium box near second anchor
        [32., 32., 8., 8.]   # Large box near third anchor
    ])
    
    reg_max = 16
    
    # Convert boxes to distances
    target_dist = bbox2dist(anchors, boxes, reg_max)
    print(f"Target distances shape: {target_dist.shape}")
    print(f"Distance range: {target_dist.min():.2f} to {target_dist.max():.2f}")
    
    # Convert distances back to boxes
    pred_boxes = dist2bbox(target_dist, anchors)
    print(f"\nOriginal boxes:\n{boxes}")
    print(f"\nRecovered boxes:\n{pred_boxes}")
    
    # Compute error
    error = torch.abs(boxes - pred_boxes).mean()
    print(f"\nMean reconstruction error: {error:.4f}")

def visualize_anchors_and_targets(anchors, targets, predictions=None):
    """Visualize anchors, targets, and predictions"""
    plt.figure(figsize=(10, 10))
    
    # Plot anchors
    plt.scatter(anchors[:, 0], anchors[:, 1], c='blue', alpha=0.1, label='Anchors')
    
    # Plot targets
    for box in targets:
        x, y, w, h = box
        plt.gca().add_patch(plt.Rectangle(
            (x - w/2, y - h/2), w, h,
            fill=False, color='green', label='Target'
        ))
    
    # Plot predictions if provided
    if predictions is not None:
        for box in predictions:
            x, y, w, h = box
            plt.gca().add_patch(plt.Rectangle(
                (x - w/2, y - h/2), w, h,
                fill=False, color='red', label='Prediction'
            ))
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def test_anchor_generation():
    """Test anchor generation"""
    print("\nTesting Anchor Generation...")
    
    # Create dummy feature maps
    feats = [
        torch.randn(1, 64, 80, 80),  # P3
        torch.randn(1, 128, 40, 40), # P4
        torch.randn(1, 256, 20, 20)  # P5
    ]
    
    strides = torch.tensor([8., 16., 32.])
    
    # Generate anchors
    anchor_points, stride_tensor = make_anchors(feats, strides)
    
    print("Anchor statistics:")
    print(f"Total anchors: {len(anchor_points)}")
    print(f"X range: {anchor_points[:, 0].min():.1f} to {anchor_points[:, 0].max():.1f}")
    print(f"Y range: {anchor_points[:, 1].min():.1f} to {anchor_points[:, 1].max():.1f}")
    print(f"Stride values: {torch.unique(stride_tensor)}")
    
    # Visualize anchors
    plt.figure(figsize=(10, 10))
    plt.scatter(anchor_points[:, 0], anchor_points[:, 1], c=stride_tensor, 
                alpha=0.1, cmap='viridis')
    plt.colorbar(label='Stride')
    plt.title('Anchor Points Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


def test_loss_computation():
    """Test the detection loss computation"""
    print("\nTesting Loss Computation...")
    
    # Create dummy predictions (batch_size=2, 3 feature levels)
    predictions = [
        torch.randn(2, 116, 4, 8, 8),  # P3
        torch.randn(2, 116, 4, 4, 4),  # P4
        torch.randn(2, 116, 4, 2, 2)   # P5
    ]


    print("Prediction stats:")
    for i, pred in enumerate(predictions):
        print(f"\nLevel {i}:")
        print(f"Mean: {pred.mean():.4f}")
        print(f"Std: {pred.std():.4f}")
        print(f"Max: {pred.max():.4f}")
        print(f"Min: {pred.min():.4f}")
        
        # Check classification logits specifically
        cls_logits = pred[:, :80]  # First 80 channels are class logits
        print(f"Class logits mean: {cls_logits.mean():.4f}")
        print(f"Class logits std: {cls_logits.std():.4f}")

    # Create dummy targets (B, max_targets, 4)
    target_boxes = torch.tensor([
        [[100., 100., 20., 20.],
         [200., 200., 40., 40.],
         [0., 0., 0., 0.]],  # padding
        [[150., 150., 30., 30.],
         [250., 250., 50., 50.],
         [0., 0., 0., 0.]]   # padding
    ])
    
    # Create dummy categories (B, max_targets)
    target_cls = torch.tensor([
        [1, 2, -1],  # -1 for padding
        [1, 3, -1]   # -1 for padding
    ])
    
    # Initialize loss function
    loss_fn = DetectionLoss(reg_max=9, nc=80)
    
    # Make anchors for each feature level
    strides = torch.tensor([8., 16., 32.])
    all_anchors = {}
    
    # Generate feature level anchors
    for level_idx, pred in enumerate(predictions):
        _, _, _, H, W = pred.shape
        stride = strides[level_idx]
        
        # Generate grid coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        
        # Create anchor points
        grid_xy = torch.stack([
            (grid_x + 0.5) * stride,
            (grid_y + 0.5) * stride
        ], dim=-1).reshape(-1, 2)
        
        all_anchors[level_idx] = grid_xy
        
        print(f"\nFeature level {level_idx}:")
        print(f"Feature map size: {H}x{W}")
        print(f"Number of anchors: {len(grid_xy)}")
        print(f"Anchor range: {grid_xy.min():.1f} to {grid_xy.max():.1f}")
    
    # Compute loss
    try:
        box_loss, dfl_loss, cls_loss, quat_loss = loss_fn(
            outputs=predictions,
            anchor_points=all_anchors,
            target_bboxes=target_boxes,
            target_categories=target_cls,
            target_scores_sum=torch.tensor([4.], device=target_boxes.device)
        )
        
        print("\nLoss computation successful!")
        print(f"Box loss: {box_loss.item():.4f}")
        print(f"DFL loss: {dfl_loss.item():.4f}")
        print(f"Class loss: {cls_loss.item():.4f}")
        # print(f"Quaternion loss: {quat_loss.item():.4f}")
        
    except Exception as e:
        print("\nError in loss computation:")
        print(str(e))
        raise e

def test_box_conversion_precision():
    """Test box conversion with various cases."""
    # Test cases
    anchors = torch.tensor([
        [10., 10.],  # Small values
        [20., 20.],  # Medium values
        [100., 100.],  # Large values
        [150., 150.]  # Very large values
    ])
    
    boxes = torch.tensor([
        [12., 12., 4., 4.],     # Small box, small offset
        [22., 22., 8., 8.],     # Medium box, small offset
        [105., 105., 20., 20.], # Large box, medium offset
        [160., 160., 40., 40.]  # Large box, large offset
    ])
    
    print("\nTesting box conversion precision...")
    print(f"Original boxes:\n{boxes}")
    
    # Convert to distances
    reg_max = 16
    distances = bbox2dist(anchors, boxes, reg_max=reg_max)
    print(f"\nDistance shape: {distances.shape}")
    print(f"Distance stats:")
    print(f"- Range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"- Mean: {distances.mean():.4f}")
    print(f"- Std: {distances.std():.4f}")
    
    # Convert back to boxes
    recovered = dist2bbox(distances, anchors)
    print(f"\nRecovered boxes:\n{recovered}")
    
    # Compute various error metrics
    abs_error = torch.abs(recovered - boxes)
    rel_error = abs_error / boxes
    max_error = abs_error.max().item()
    
    print("\nAbsolute errors:")
    print(f"Center X: {abs_error[:, 0].mean():.6f} (max: {abs_error[:, 0].max():.6f})")
    print(f"Center Y: {abs_error[:, 1].mean():.6f} (max: {abs_error[:, 1].max():.6f})")
    print(f"Width: {abs_error[:, 2].mean():.6f} (max: {abs_error[:, 2].max():.6f})")
    print(f"Height: {abs_error[:, 3].mean():.6f} (max: {abs_error[:, 3].max():.6f})")
    
    print("\nRelative errors:")
    print(f"Center X: {rel_error[:, 0].mean():.6f} (max: {rel_error[:, 0].max():.6f})")
    print(f"Center Y: {rel_error[:, 1].mean():.6f} (max: {rel_error[:, 1].max():.6f})")
    print(f"Width: {rel_error[:, 2].mean():.6f} (max: {rel_error[:, 2].max():.6f})")
    print(f"Height: {rel_error[:, 3].mean():.6f} (max: {rel_error[:, 3].max():.6f})")
    
    return max_error



if __name__ == "__main__":
    # Run tests
    test_box_conversion_precision()
    test_distance_conversion()
    test_loss_computation()
    # test_anchor_generation()