# engine/validator.py

import torch
from tqdm import tqdm
import os
from loss.box_loss import DetectionLoss

class Validator:
    def __init__(self, model, dataloader, metric, device, save_dir='runs/validate'):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.metric = metric
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Add metric logging
        self.log_file = os.path.join(save_dir, 'validation_metrics.txt')

    def validate(self, loss_fn):
        """
        Enhanced validation with proper quaternion handling and metrics.
        """
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'dfl_loss': 0.0,
            'quat_loss': 0.0,
            'map50': 0.0,
            'map75': 0.0
        }
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), 
                              desc="Validation", leave=False)
            
            for batch_idx, batch in progress_bar:
                # Move data to device
                images = batch['image'].to(self.device, non_blocking=True)
                target_bboxes = batch['bbox'].to(self.device, non_blocking=True)
                target_categories = batch['category'].to(self.device, non_blocking=True)
                
                # Forward pass with quaternion structure preservation
                pred_dist, pred_scores = self.model(images)  # [B, C, 4, H, W]
                
                # Make anchors for detection
                anchors = self.model.make_anchors(pred_dist.shape[-2:], pred_dist.device)
                
                # Compute losses
                box_loss, dfl_loss, quat_loss = loss_fn(
                    pred_dist=pred_dist,
                    pred_scores=pred_scores,
                    anchor_points=anchors,
                    target_bboxes=target_bboxes,
                    target_categories=target_categories,
                    target_scores_sum=target_categories.sum()
                )
                
                # Update running metrics
                total_loss = box_loss + dfl_loss
                if quat_loss is not None:
                    total_loss += quat_loss
                
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['box_loss'] += box_loss.item()
                val_metrics['dfl_loss'] += dfl_loss.item()
                if quat_loss is not None:
                    val_metrics['quat_loss'] += quat_loss.item()
                
                # Store predictions and targets for mAP calculation
                predictions.append(self.process_predictions(pred_dist, pred_scores, anchors))
                targets.append({
                    'boxes': target_bboxes,
                    'labels': target_categories
                })
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}"
                })
        
        # Calculate mAP
        map_metrics = self.metric.compute_map(predictions, targets)
        val_metrics.update(map_metrics)
        
        # Normalize metrics
        num_batches = len(self.dataloader)
        for k in ['total_loss', 'box_loss', 'dfl_loss', 'quat_loss']:
            val_metrics[k] /= num_batches
        
        # Log metrics
        self._log_metrics(val_metrics)
        
        return val_metrics
    
    def process_predictions(self, pred_dist, pred_scores, anchors):
        """Process raw predictions into format for mAP calculation."""
        # Decode boxes
        pred_bboxes = self.model.bbox_decode(pred_dist, anchors)
        
        # Get scores and labels
        scores, labels = pred_scores.max(dim=1)
        
        return {
            'boxes': pred_bboxes,
            'scores': scores,
            'labels': labels
        }
    
    def _log_metrics(self, metrics):
        """Log validation metrics to file."""
        with open(self.log_file, 'a') as f:
            f.write(f"\nValidation Metrics:")
            for k, v in metrics.items():
                f.write(f"\n{k}: {v:.4f}")
            f.write("\n" + "-"*50)
