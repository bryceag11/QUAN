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
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, 
                 scaler, loss_fn, metrics, device, save_dir, grad_clip_val=0.5, 
                 visualize=True, vis_batch_freq=100):
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
        super().__init__()
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
        
        # Enhanced logging
        self.log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.train_log = os.path.join(self.log_dir, 'train_log.txt')
        
        # Initialize validation metrics
        self.best_map = 0.0
        self.best_epoch = 0

    def validate(self):
        """Run validation with mAP calculation."""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0
        }
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), 
                      desc="Validation", leave=False)
            
            for batch_idx, batch in pbar:
                images = batch['image'].to(self.device)
                target_boxes = batch['bbox'].to(self.device)
                target_classes = batch['category'].to(self.device)
                
                # Forward pass
                pred_cls, pred_box = self.model(images)
                
                # Compute losses
                loss_cls, loss_box, total_loss = self.loss_fn(
                    pred_cls, pred_box,
                    {'boxes': target_boxes, 'labels': target_classes}
                )
                
                # Update running metrics
                val_metrics['total_loss'] += total_loss.item()
                val_metrics['box_loss'] += loss_box.item()
                val_metrics['cls_loss'] += loss_cls.item()
                
                # Get predictions for mAP calculation
                for level_idx in range(len(pred_cls)):
                    boxes, classes, scores = get_predictions(
                        pred_cls[level_idx], 
                        pred_box[level_idx]
                    )
                    predictions.append({
                        'boxes': boxes,
                        'scores': scores,
                        'labels': classes
                    })
                    targets.append({
                        'boxes': target_boxes,
                        'labels': target_classes
                    })

                pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # Compute mAP
        self.metrics.process(predictions, targets)
        map_metrics = self.metrics.results_dict

        # Normalize metrics
        num_batches = len(self.val_dataloader)
        for k in ['total_loss', 'box_loss', 'cls_loss']:
            val_metrics[k] /= num_batches

        # Combine all metrics
        val_metrics.update(map_metrics)
        
        return val_metrics

    def train_one_epoch(self, epoch):
        """Train for one epoch with complete loss handling."""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
        }
        
        # Added num_samples counter
        num_samples = 0
        running_loss = 0.0
        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), 
                desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in pbar:
            try:
                # Clear memory and gradients
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                
                images = batch['image'].to(self.device)
                targets = {
                    'boxes': batch['bbox'].to(self.device),
                    'labels': batch['category'].to(self.device)
                }
                

                # Forward pass with autocast
                with torch.amp.autocast('cuda'):
                    pred_cls, pred_reg, anchors = self.model(images)
                    total_loss, num_pos = self.loss_fn(pred_cls, pred_reg, targets, anchors)

                # Backward pass
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                # Optimize
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Update metrics
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['box_loss'] += box_loss.item()
                epoch_metrics['cls_loss'] += cls_loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'box': f"{box_loss.item():.4f}",
                    'cls': f"{cls_loss.item():.4f}"
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}:")
                print(str(e))
                continue


                
        num_batches = len(self.train_dataloader)
        epoch_metrics = {k: v / num_batches for k, v in epoch_metrics.items()}
        
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

    def validate_and_save(self, epoch):
        """Run validation and save best model."""
        # Run validation
        val_metrics = self.validate()
        
        # Save if best mAP
        if val_metrics['map50'] > self.best_map:
            self.best_map = val_metrics['map50']
            self.best_epoch = epoch
            self.save_checkpoint(epoch, is_best=True)
        
        # Regular checkpoint
        if epoch % 10 == 0:
            self.save_checkpoint(epoch)
        
        return val_metrics

    def save_checkpoint(self, epoch, is_best=False):
        """Save enhanced checkpoint with quaternion state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'best_map': self.best_map
        }
        
        # Save checkpoint
        save_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, save_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def visualize_batch(self, images, pred_dist, pred_scores, target_bboxes, 
                       target_categories, batch_idx, epoch):
        """Visualize batch with quaternion predictions."""
        from utils.visualization import visualize_quaternion_predictions
        
        save_dir = os.path.join(self.save_dir, 'visualizations', f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        visualize_quaternion_predictions(
            images=images.cpu(),
            pred_dist=pred_dist.detach().cpu(),
            pred_scores=pred_scores.detach().cpu(),
            target_bboxes=target_bboxes.cpu(),
            target_categories=target_categories.cpu(),
        )

    def _log_metrics(self, epoch, metrics):
        """Log training metrics."""
        with open(self.train_log, 'a') as f:
            f.write(f"\nEpoch {epoch}:")
            for k, v in metrics.items():
                f.write(f"\n{k}: {v:.4f}")
            f.write("\n" + "-"*50)
