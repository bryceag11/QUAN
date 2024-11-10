# train.py

#!/usr/bin/env python
# train.py

import torch
import yaml
import argparse
from data.dataloader import get_quaternion_dataloader
from models.model_builder import load_model_from_yaml
from loss.box_loss import RotatedBboxLoss
from engine.trainer import Trainer
from engine.validator import Validator
from utils.metrics import OBBMetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train Quaternion-based YOLO Model")
    parser.add_argument('--config', type=str, default='configs/models/q.yaml', help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Directory to save checkpoints and logs')
    parser.add_argument('--profile', action='store_true', help='Profile model FLOPs and parameters')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset configuration
    with open(args.data, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    active_dataset = dataset_config['active_dataset']
    train_dataset_info = dataset_config['datasets'][active_dataset]['train']
    val_dataset_info = dataset_config['datasets'][active_dataset]['val']

    # Load model from YAML configuration
    model, nc = load_model_from_yaml(args.config)

    # Move model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Profile model if requested
    if args.profile:
        from utils.profile import get_model_complexity
        complexity = get_model_complexity(model, input_size=(4, 640, 640))  # Adjust input size as needed
        print(complexity)
        return  # Exit after profiling

    # Define transforms and augmentations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # Add normalization if needed
    ])
    augmentations = None  # Replace with your QuaternionAugmentations if applicable

    # Initialize dataloaders
    train_dataloader = get_quaternion_dataloader(
        img_dir=train_dataset_info['img_dir'],
        ann_file=train_dataset_info['ann_file'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        dataset_type=active_dataset.lower(),  # 'dota' or 'coco'
        transform=transform,
        augmentations=augmentations
    )
    val_dataloader = get_quaternion_dataloader(
        img_dir=val_dataset_info['img_dir'],
        ann_file=val_dataset_info['ann_file'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        dataset_type=active_dataset.lower(),  # 'dota' or 'coco'
        transform=transform,
        augmentations=None  # Typically, no augmentations during validation
    )

    # Initialize loss function
    loss_fn = RotatedBboxLoss(reg_max=16)  # Adjust reg_max as needed

    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Initialize metrics
    class_names = {0: 'class1', 1: 'class2'}  # Update with actual class names
    metric = OBBMetrics(save_dir=args.save_dir, plot=True, names=class_names)

    # Initialize trainer and validator
    trainer = Trainer(model, train_dataloader, optimizer, scheduler, loss_fn, device=device, save_dir=args.save_dir)
    validator = Validator(model, val_dataloader, metric, device=device, save_dir=os.path.join(args.save_dir, 'validate'))

    # Create directory for validation results
    os.makedirs(os.path.join(args.save_dir, 'validate'), exist_ok=True)

    # Start training
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        trainer.train_one_epoch(epoch)
        print(f"Epoch {epoch+1} training completed. Starting validation.")
        validator.validate()
        print(f"Epoch {epoch+1} validation completed.")
        trainer.save_checkpoint(epoch)

    # Final Visualization (if any)
    metric.plot_final_metrics(save_dir=args.save_dir, names=class_names)

    print("Training and validation completed. Metrics and visualizations saved.")

if __name__ == "__main__":
    main()
