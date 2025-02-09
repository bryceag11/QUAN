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

    def train_one_epoch(self, epoch):
        """Train for one epoch with visualization and metrics."""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'num_pos': 0
        }
        
        if epoch == 0:  # Or based on memory monitoring
            torch.cuda.empty_cache()
        
        pbar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), 
                desc=f"Epoch {epoch+1}", leave=False)
        
        # Zero gradients once before the loop if using set_to_none=True
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in pbar:
            try:
                # Move data to device - this is now non-blocking due to pin_memory=True
                images = batch['image'].to(self.device, non_blocking=True)
                target_boxes = batch['bbox'].to(self.device, non_blocking=True)
                target_labels = batch['category'].to(self.device, non_blocking=True)
                
                targets = {
                    'boxes': target_boxes,
                    'labels': target_labels
                }

                # Forward pass with autocast
                with torch.amp.autocast('cuda'):
                    pred_cls, pred_reg, anchors = self.model(images)
                    loss_dict = self.loss_fn(pred_cls, pred_reg, targets, anchors)

                # Extract loss and num_pos from dict
                total_loss = loss_dict['loss']
                num_pos = loss_dict['num_pos']

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
                
                self.optimizer.zero_grad(set_to_none=True)

                # Update metrics
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['num_pos'] += num_pos
                
                # Visualize batch
                # if self.visualize and batch_idx % self.vis_batch_freq == 0:
                #     # Concatenate predictions from all feature levels
                #     all_cls = torch.cat([p.view(p.size(0), -1, p.size(-1)) for p in pred_cls], dim=1)
                #     all_reg = torch.cat([r.view(r.size(0), -1, r.size(-1)) for r in pred_reg], dim=1)
                    
                #     self.visualize_batch(
                #         images=images,
                #         pred_dist=all_reg,
                #         pred_scores=torch.sigmoid(all_cls),
                #         target_bboxes=targets['boxes'],
                #         target_categories=targets['labels'],
                #         batch_idx=batch_idx,
                #         epoch=epoch
                #     )
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'pos': num_pos
                })
                
            except Exception as e:
                print(f"\nError in batch {batch_idx}:")
                print(str(e))
                continue

        # Average the loss over batches
        num_batches = len(self.train_dataloader)
        epoch_metrics['total_loss'] /= num_batches
        
        # Plot training curves using your existing metrics
        if hasattr(self, 'metrics') and hasattr(self.metrics, 'plot'):
            self.metrics.plot()
        
        return epoch_metrics

    def validate(self):
        """Enhanced validation with proper metrics handling."""
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'num_pos': 0
        }
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader), 
                    desc="Validation", leave=False)
            
            for batch_idx, batch in pbar:
                images = batch['image'].to(self.device)
                target_dict = {
                    'boxes': batch['bbox'].to(self.device),
                    'labels': batch['category'].to(self.device)
                }
                
                # Forward pass
                pred_cls, pred_reg, anchors = self.model(images)
                loss_dict = self.loss_fn(pred_cls, pred_reg, anchors,
                                    target_dict['labels'], target_dict['boxes'])
                
                # Update running metrics
                val_metrics['total_loss'] += loss_dict['loss'].item()
                val_metrics['num_pos'] += loss_dict['num_pos']
                
                # Store predictions and targets for metrics computation
                predictions.append({
                    'boxes': pred_reg,
                    'scores': torch.sigmoid(pred_cls),  # Apply sigmoid for scores
                    'labels': pred_cls.argmax(dim=-1)
                })
                targets.append(target_dict)
                
                pbar.set_postfix({'loss': f"{loss_dict['loss'].item():.4f}"})
        
        # Normalize metrics
        num_batches = len(self.val_dataloader)
        val_metrics['total_loss'] /= num_batches
        
        # Process metrics using your DetMetrics class
        if hasattr(self, 'metrics'):
            self.metrics.process(
                tp=predictions,  
                conf=torch.cat([p['scores'] for p in predictions]),
                pred_cls=torch.cat([p['labels'] for p in predictions]),
                target_cls=torch.cat([t['labels'] for t in targets])
            )
            val_metrics.update(self.metrics.results_dict)
        
        return val_metrics

    def visualize_batch(self, images, pred_dist, pred_scores, target_bboxes, target_categories, batch_idx, epoch):
        """
        Visualize batch with predictions.

        Args:
            images (torch.Tensor): Batch of images
            pred_dist (torch.Tensor): Predicted distributions
            pred_scores (torch.Tensor): Predicted scores
            target_bboxes (torch.Tensor): Target bounding boxes
            target_categories (torch.Tensor): Target category labels
            batch_idx (int): Batch index
            epoch (int): Current epoch
        """
        save_dir = os.path.join(self.save_dir, 'visualizations', f'epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        from utils.visualization import plot_batch_predictions
        
        # Convert predictions to appropriate format
        predictions = {
            'pred_dist': pred_dist,
            'pred_scores': pred_scores,
        }
        
        targets = {
            'boxes': target_bboxes,
            'category': target_categories
        }
        
        # Call plot_batch_predictions with correct arguments
        plot_batch_predictions(
            images=images,
            predictions=predictions,
            targets=targets,
            save_dir=save_dir,
            batch_idx=batch_idx
        )

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

    def _log_metrics(self, epoch, metrics):
        """Log training metrics."""
        with open(self.train_log, 'a') as f:
            f.write(f"\nEpoch {epoch}:")
            for k, v in metrics.items():
                f.write(f"\n{k}: {v:.4f}")
            f.write("\n" + "-"*50)
