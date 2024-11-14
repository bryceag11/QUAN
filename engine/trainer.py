# engine/trainer.py
import gc
import os
import csv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from loss.box_loss import DetectionLoss  # Ensure DetectionLoss is imported
from torch.cuda.amp import autocast
from utils.ops import make_anchors
import math 


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, scaler, loss_fn, metrics, 
                 device, save_dir, grad_clip_val=1.0, visualize=True, vis_batch_freq=100):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_dataloader (DataLoader): DataLoader for the training data.
            val_dataloader (DataLoader): DataLoader for the validation data.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
            loss_fn (DetectionLoss): Loss function to use (DetectionLoss).
            metrics (MetricsClass): Instance of a metrics class to compute evaluation metrics.
            device (str): Device to train on ('cuda' or 'cpu').
            save_dir (str): Directory to save training logs and plots.
            grad_clip_val (float, optional): Maximum gradient norm for clipping. Defaults to 1.0.
            visualize (bool, optional): Whether to visualize training progress. Defaults to True.
            vis_batch_freq (int, optional): Frequency (in batches) to visualize training progress. Defaults to 100.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device
        self.save_dir = save_dir
        self.grad_clip_val = grad_clip_val
        self.visualize = visualize
        self.vis_batch_freq = vis_batch_freq
        
        os.makedirs(save_dir, exist_ok=True)
        self.csv_file = os.path.join(save_dir, 'training_log.csv')
        self._initialize_csv()


    def train_one_epoch(self, epoch):
        """Train for one epoch with proper mixed precision handling and NaN/Infty checks."""
        self.model.train()
        progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), 
                            desc=f"Epoch {epoch+1}", leave=False)
        
        epoch_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'dfl_loss': 0.0,
            'quat_loss': 0.0
        }
        
        for batch_idx, batch in progress_bar:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            target_bboxes = batch['bbox'].to(self.device, non_blocking=True)
            target_categories = batch['category'].to(self.device, non_blocking=True)
            
            # Memory after data loading
            allocated_mem = torch.cuda.memory_allocated(device=self.device) / 1e6
            reserved_mem = torch.cuda.memory_reserved(device=self.device) / 1e6
            print(f"[Batch {batch_idx}] After data loading:")
            print(f"Allocated: {allocated_mem:.2f} MB")
            print(f"Reserved: {reserved_mem:.2f} MB")
            
            try:
                # # Forward pass with autocast
                # with torch.amp.autocast("cuda"):
                    preds = self.model(images)  # [B, C, 4, H, W]
                    pred = preds[0]  # Assuming preds is a list or tuple, adjust if necessary
                    
                    B, C, D, H, W = pred.shape
                    num_anchors = H * W
                    
                    # Reshape predictions
                    pred = pred.permute(0, 3, 4, 1, 2).reshape(B, H*W, -1)  # [B, H*W, C*D]
                    reg_max = 16
                    pred_dist = pred[..., :reg_max * 4]  # [B, H*W, 64]
                    
                    # Clamp pred_dist to prevent extreme values before decoding
                    pred_dist = torch.clamp(pred_dist, min=-10.0, max=10.0)
                    
                    # Generate anchor points
                    anchor_points = self.loss_fn.make_anchors(H, W, pred.device)  # [num_anchors, 2]
                    
                    # Decode predictions to bounding boxes
                    pred_bboxes = self.loss_fn.bbox_decode(pred_dist, anchor_points)  # [B, num_anchors, 4]
                    
                    # Clamp decoded_bboxes to ensure they are within image boundaries
                    pred_bboxes = torch.clamp(pred_bboxes, min=0.0, max=1e4)
                    
                    # Check for NaNs or Infs in preds and pred_bboxes
                    if torch.isnan(pred).any() or torch.isinf(pred).any():
                        print(f"\nError: NaNs or Infinities detected in preds at batch {batch_idx}. Aborting training.")
                        raise ValueError("NaNs or Infs in preds.")
                    
                    if torch.isnan(pred_bboxes).any() or torch.isinf(pred_bboxes).any():
                        print(f"\nError: NaNs or Infinities detected in pred_bboxes at batch {batch_idx}. Aborting training.")
                        raise ValueError("NaNs or Infs in pred_bboxes.")
                    
                    # Prepare targets
                    expanded_target_boxes = []
                    fg_mask = torch.zeros(B, num_anchors, dtype=torch.bool, device=self.device)
                    
                    for b in range(B):
                        valid_targets = target_categories[b] > 0
                        num_valid = valid_targets.sum()
                        
                        batch_boxes = torch.zeros(num_anchors, 4, device=self.device)
                        if num_valid > 0:
                            batch_boxes[:num_valid] = target_bboxes[b, :num_valid]
                            fg_mask[b, :num_valid] = True
                            
                        expanded_target_boxes.append(batch_boxes)
                    
                    expanded_target_boxes = torch.stack(expanded_target_boxes)  # [B, num_anchors, 4]
                    
                    # Create target scores based on foreground mask
                    target_scores = torch.zeros(B, num_anchors, device=self.device)
                    target_scores[fg_mask] = 1.0
                    target_scores_sum = target_scores.sum().clamp(min=1)  # Avoid division by zero
                    
                    # Compute losses
                    box_loss, dfl_loss, quat_loss = self.loss_fn(
                        pred_dist=pred_dist,
                        pred_bboxes=pred_bboxes,
                        anchor_points=anchor_points,
                        target_bboxes=expanded_target_boxes,
                        target_scores=target_scores,
                        target_scores_sum=target_scores_sum,
                        fg_mask=fg_mask
                    )
                    
                    # Mean losses
                    box_loss = box_loss.mean()
                    dfl_loss = dfl_loss.mean()
                    if quat_loss is not None:
                        quat_loss = quat_loss.mean()
                    
                    # Total loss
                    total_loss = box_loss + dfl_loss
                    if quat_loss is not None:
                        total_loss += quat_loss
                    
                    # Check for non-finite losses
                    if not torch.isfinite(total_loss):
                        print(f"\nError: Non-finite loss detected at batch {batch_idx}!")
                        print(f"box_loss: {box_loss.item()}")
                        print(f"dfl_loss: {dfl_loss.item()}")
                        if quat_loss is not None:
                            print(f"quat_loss: {quat_loss.item()}")
                        print(f"pred_bboxes stats: min={pred_bboxes.min().item()}, max={pred_bboxes.max().item()}")
                        print(f"target_boxes stats: min={expanded_target_boxes.min().item()}, max={expanded_target_boxes.max().item()}")
                        
                        raise ValueError("Non-finite loss detected.")
            
            except (RuntimeError, ValueError) as e:
                print(f"\nCritical Error in batch {batch_idx}: {e}")
                print("Aborting training to prevent memory leaks.")
                raise e
            
            # Backward pass and optimizer step
            try:
                # Scale and backward
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                if self.grad_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Memory after optimizer step
                allocated_mem = torch.cuda.memory_allocated(device=self.device) / 1e6
                reserved_mem = torch.cuda.memory_reserved(device=self.device) / 1e6
                print(f"[Batch {batch_idx}] After optimizer step:")
                print(f"Allocated: {allocated_mem:.2f} MB")
                print(f"Reserved: {reserved_mem:.2f} MB")
                
                # Update scheduler
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update metrics
                with torch.no_grad():
                    epoch_metrics['total_loss'] += total_loss.item()
                    epoch_metrics['box_loss'] += box_loss.item()
                    epoch_metrics['dfl_loss'] += dfl_loss.item()
                    if quat_loss is not None:
                        epoch_metrics['quat_loss'] += quat_loss.item()
                
                # Update progress bar every 10 batches
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        'loss': f"{total_loss.item():.4f}",
                        'box': f"{box_loss.item():.4f}",
                        'dfl': f"{dfl_loss.item():.4f}",
                    })
                    progress_bar.refresh()
            
            except RuntimeError as e:
                print(f"\nError during backward/optimizer step in batch {batch_idx}:")
                print(f"Error type: {type(e)}")
                print(f"Error message: {str(e)}")
                print("\nCUDA Memory Status:")
                print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
                print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
                raise e
            
            # Clean up to prevent memory leaks
            del preds, pred, pred_dist, pred_bboxes, anchor_points, expanded_target_boxes, total_loss, quat_loss, dfl_loss, box_loss
            torch.cuda.empty_cache()
            gc.collect()
            
        return epoch_metrics

    def _initialize_csv(self):
        """Initialize the CSV log file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'total_loss', 'box_loss', 'dfl_loss', 
                            'quat_loss'])

    def _log_to_csv(self, epoch, batch, total_loss, box_loss, dfl_loss, quat_loss):
        """Log the training metrics to CSV."""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                batch,
                total_loss,
                box_loss,
                dfl_loss,
                quat_loss
            ])

    def visualize_batch(self, images, preds, target_bboxes, target_categories, batch_idx, epoch):
        """Visualize a batch of predictions."""
        vis_dir = os.path.join(self.save_dir, 'visualizations', f'epoch_{epoch}')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Take first image from batch
        with torch.no_grad():
            img = images[0].cpu()
            pred_tensor = preds[0].cpu()  # [1, 148, 4, 18, 18]
            batch_size, num_anchors, channels, H, W = pred_tensor.shape
            
            # Example extraction based on channel mapping
            # Adjust indices based on your model
            pred_obj = pred_tensor[:, :, 0, :, :].unsqueeze(-1)  # [1, 148, 1, H, W]
            pred_cls = pred_tensor[:, :, 1, :, :].unsqueeze(-1)  # [1, 148, 1, H, W]
            pred_bbox = pred_tensor[:, :, 2:4, :, :].permute(0, 1, 3, 4, 2).reshape(batch_size, num_anchors, -1)  # [1, 148, 2]
            
            if channels >= 8:
                pred_quat = pred_tensor[:, :, 4:8, :, :].permute(0, 1, 3, 4, 2).reshape(batch_size, num_anchors, -1)  # [1, 148, 4]
            else:
                pred_quat = None
            
            # Convert image for plotting
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 4:  # If quaternion image
                img_np = img_np[:, :, :3]  # Take only RGB channels
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Plot image
            ax.imshow(img_np)
            ax.axis('off')
            
            # Plot predicted boxes
            for box in pred_bbox[0].numpy():
                rect = plt.Rectangle(
                    (box[0], box[1]),
                    box[2] - box[0],
                    box[3] - box[1],
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)
            
            # Plot ground truth boxes
            for i, box in enumerate(target_bboxes[0].cpu().numpy()):
                if target_categories[0][i] > 0:  # Assuming category 0 is background
                    rect = plt.Rectangle(
                        (box[0], box[1]),
                        box[2] - box[0],
                        box[3] - box[1],
                        fill=False,
                        edgecolor='green',
                        linewidth=2
                    )
                    ax.add_patch(rect)
            
            # Optionally, plot quaternions or other annotations
            
            plt.savefig(os.path.join(vis_dir, f'batch_{batch_idx}.png'))
            plt.close(fig)

    def save_checkpoint(self, epoch):
        """Save a checkpoint of the model and training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': self.metrics.get_current_metrics() if hasattr(self.metrics, 'get_current_metrics') else None
        }
        torch.save(checkpoint, os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    def validate(self):
        """Implement validation logic here."""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'dfl_loss': 0.0,
            'quat_loss': 0.0
        }
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), desc="Validation", leave=False)
            for batch_idx, batch in progress_bar:
                images = batch['image'].to(self.device, non_blocking=True)
                target_bboxes = batch['bbox'].to(self.device, non_blocking=True)
                target_categories = batch['category'].to(self.device, non_blocking=True)
                target_quat = batch.get('quat')  # Assuming quaternions are provided in the batch
                if target_quat is not None:
                    target_quat = target_quat.to(self.device, non_blocking=True)
                
                preds = self.model(images)
                
                # Ensure preds is a list with at least one tensor
                if not isinstance(preds, list) or len(preds) == 0:
                    raise ValueError("Model output 'preds' must be a non-empty list of tensors.")
                
                pred_tensor = preds[0]  # Extract the first tensor from the list
                batch_size, num_anchors, channels, H, W = pred_tensor.shape
                
                # Example channel mapping
                pred_obj = pred_tensor[:, :, 0, :, :].unsqueeze(-1)  # [batch_size, num_anchors, 1, H, W]
                pred_cls = pred_tensor[:, :, 1, :, :].unsqueeze(-1)  # [batch_size, num_anchors, 1, H, W]
                pred_bbox = pred_tensor[:, :, 2:4, :, :].permute(0, 1, 3, 4, 2).reshape(batch_size, num_anchors, -1)  # [batch_size, num_anchors, 2]
                
                if channels >= 8:
                    pred_quat = pred_tensor[:, :, 4:8, :, :].permute(0, 1, 3, 4, 2).reshape(batch_size, num_anchors, -1)  # [batch_size, num_anchors, 4]
                else:
                    pred_quat = None
                
                # Flatten spatial dimensions if DetectionLoss expects flat predictions
                pred_obj_flat = pred_obj.reshape(batch_size, num_anchors, -1)  # [batch_size, num_anchors, H*W]
                pred_cls_flat = pred_cls.reshape(batch_size, num_anchors, -1)  # [batch_size, num_anchors, H*W]
                pred_bbox_flat = pred_bbox.reshape(batch_size, num_anchors, -1)  # [batch_size, num_anchors, 2]
                if pred_quat is not None:
                    pred_quat_flat = pred_quat.reshape(batch_size, num_anchors, -1)  # [batch_size, num_anchors, 4]
                else:
                    pred_quat_flat = None
                
                # Combine objectness and class scores for pred_dist if DetectionLoss expects it
                pred_dist = torch.cat([pred_obj_flat, pred_cls_flat], dim=-1)  # [batch_size, num_anchors, 1 + 1 * H * W]
                
                # Generate or retrieve anchor points
                anchor_points = self.loss_fn.assigner.make_anchors(H, W, self.device)  # [num_anchors, 2]
                
                # Assign targets to anchors using the assigner
                fg_mask, target_bboxes_aligned, target_categories_aligned = self.loss_fn.assigner.assign(
                    pred_bboxes=pred_bbox_flat, 
                    pred_scores=pred_cls_flat, 
                    target_bboxes=target_bboxes, 
                    target_categories=target_categories
                )
                
                # Compute loss
                box_loss, dfl_loss, quat_loss = self.loss_fn(
                    pred_dist=pred_dist, 
                    pred_bboxes=pred_bbox_flat, 
                    anchor_points=anchor_points, 
                    target_bboxes=target_bboxes_aligned, 
                    target_scores=batch['obj'], 
                    target_scores_sum=batch['obj'].sum(), 
                    fg_mask=fg_mask, 
                    pred_quat=pred_quat_flat, 
                    target_quat=target_quat
                ).detach()
                
                # Total loss
                total_loss = box_loss + dfl_loss
                if quat_loss is not None:
                    total_loss += quat_loss
                
                # Update metrics
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['box_loss'] += box_loss.item()
                val_metrics['dfl_loss'] += dfl_loss.item()
                if quat_loss is not None:
                    val_metrics['quat_loss'] += quat_loss.item()
        
        # After the validation loop
        # Normalize metrics
        num_batches = max(len(self.val_dataloader), 1)
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        # Optionally, log or save validation metrics
        print(f"Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        return val_metrics
