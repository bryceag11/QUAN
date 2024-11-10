# engine/trainer.py

import torch
from tqdm import tqdm
import csv
import os

class Trainer:
    def __init__(self, model, dataloader, optimizer, scheduler, loss_fn, device, save_dir):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            dataloader (DataLoader): DataLoader for the training data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            loss_fn (callable): Loss function.
            device (str): Device to train on ('cuda' or 'cpu').
            save_dir (str): Directory to save checkpoints and logs.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.csv_file = os.path.join(save_dir, 'training_log.csv')
        self._initialize_csv()

    def _initialize_csv(self):
        """
        Initialize the CSV file for logging training metrics.
        """
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'batch', 'loss', 'bbox_loss', 'cls_loss', 'quat_loss'])

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.
        """
        self.model.train()
        progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, batch in progress_bar:
            images = batch['image'].to(self.device)         # Shape: (B, 4, H, W)
            target_bboxes = batch['bbox'].to(self.device)   # Shape: (B, 4)
            target_categories = batch['category'].to(self.device)  # Shape: (B,)
            target_quats = batch['quat'].to(self.device)    # Shape: (B, 4)
            anchor_points = batch.get('anchor_points', None)  # Shape: (B, 2) or other, depending on implementation

            # Forward pass
            preds = self.model(images)  # Adjust based on your model's forward method

            # Compute loss
            loss, losses = self.loss_fn(preds, target_bboxes, target_categories, target_quats, anchor_points)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            # Log to CSV
            self._log_to_csv(epoch+1, batch_idx+1, losses)

    def _log_to_csv(self, epoch, batch, losses):
        """
        Log training metrics to CSV.

        Args:
            epoch (int): Current epoch number.
            batch (int): Current batch number.
            losses (dict): Dictionary containing individual loss components.
        """
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                batch,
                losses.get('total_loss', 0.0),
                losses.get('bbox_dist_loss', 0.0),
                losses.get('cls_loss', 0.0),
                losses.get('quat_loss', 0.0)
            ])

    def save_checkpoint(self, epoch):
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
