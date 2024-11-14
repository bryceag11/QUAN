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

    def validate(self, loss_fn):
        """
        Perform validation on the validation dataset.

        Args:
            loss_fn (nn.Module): The loss function used during training (BBoxLoss or RotatedBBoxLoss).
        """
        self.model.eval()
        val_metrics = {
            'total_loss': 0.0,
            'box_loss': 0.0,
            'cls_loss': 0.0,
            'obj_loss': 0.0,
            'quat_reg_loss': 0.0,
            'smooth_loss': 0.0
        }
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Validation", leave=False)
            for batch_idx, batch in progress_bar:
                images = batch['image'].to(self.device, non_blocking=True)
                target_bboxes = batch['bbox'].to(self.device, non_blocking=True)
                target_categories = batch['category'].to(self.device, non_blocking=True)
                target_quat = batch.get('quat')  # Assuming quaternions are provided in the batch
                if target_quat is not None:
                    target_quat = target_quat.to(self.device, non_blocking=True)
                
                preds = self.model(images)
                
                # Compute loss
                if isinstance(loss_fn, RotatedBBoxLoss):
                    # Assuming preds include both bbox predictions and quaternion predictions
                    pred_dist, pred_bboxes_pred, pred_quat_pred = preds  # Adjust based on your model's output
                    loss_dict = loss_fn(
                        pred_dist=pred_dist,
                        target_bboxes=target_bboxes,
                        target_categories=target_categories,
                        pred_quat=pred_quat_pred,
                        target_quat=target_quat
                    )
                else:
                    loss_dict = loss_fn(preds, target_bboxes, target_categories)
                
                # Update metrics
                for k, v in loss_dict.items():
                    if k in val_metrics:
                        val_metrics[k] += v.item()

        # Normalize metrics
        num_batches = max(len(self.dataloader), 1)
        val_metrics = {k: v / num_batches for k, v in val_metrics.items()}
        
        # Optionally, log or save validation metrics
        # For example, append to a CSV or update a metrics object
        
        print(f"Validation Metrics:")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Save validation metrics if needed
        # self.metric.update(val_metrics)
        
        return val_metrics
