# engine/validator.py

import torch
from tqdm import tqdm
import os

class Validator:
    def __init__(self, model, dataloader, metric, device, save_dir='runs/validate'):
        """
        Initialize the Validator.

        Args:
            model (nn.Module): The PyTorch model to evaluate.
            dataloader (DataLoader): DataLoader for the validation data.
            metric (MetricsClass): Instance of a metrics class to compute evaluation metrics.
            device (str): Device to evaluate on ('cuda' or 'cpu').
            save_dir (str, optional): Directory to save validation logs and plots.
        """
        self.model = model.to(device)
        self.dataloader = dataloader
        self.metric = metric
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def validate(self):
        """
        Perform validation on the validation dataset.
        """
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Validation", leave=False)
            for batch_idx, batch in progress_bar:
                images = batch['image'].to(self.device)         # Shape: (B, 4, H, W)
                target_bboxes = batch['bbox'].to(self.device)   # Shape: (B, N, 4)
                target_categories = batch['category'].to(self.device)  # Shape: (B, N)
                target_quats = batch['quat'].to(self.device)    # Shape: (B, N, 4)
                bbox_types = batch['bbox_type'].to(self.device) # Shape: (B, N)
                
                # Reshape targets to (B*N, ...)
                B, N = target_bboxes.shape[:2]
                target_bboxes = target_bboxes.view(-1, 4)        # Shape: (B*N, 4)
                target_categories = target_categories.view(-1)    # Shape: (B*N,)
                target_quats = target_quats.view(-1, 4)          # Shape: (B*N, 4)
                bbox_types = bbox_types.view(-1)                # Shape: (B*N,)
                
                # Forward pass
                preds = self.model(images)  # Adjust based on your model's forward method, expected shape (B, C)
                
                # Compute metrics
                self.metric.update(preds, target_bboxes, target_categories, target_quats, bbox_types)
        
        # Compute final metrics
        self.metric.compute()
        
        # Generate visualizations
        self.metric.visualize(save_dir=self.save_dir)
        
        print(f"Validation completed. Metrics and visualizations saved to {self.save_dir}")
