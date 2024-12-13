# validate.py

#!/usr/bin/env python
# validate.py

import torch
import yaml
import argparse
from data.dataloader import get_quaternion_dataloader
from models.model_builder import load_model_from_yaml
from loss.box_loss import RotatedBboxLoss  # Optional, if needed for validation
from engine.validator import Validator
from utils.metrics import OBBMetrics
from torchvision import transforms
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Validate Quaternion-based YOLO Model")
    parser.add_argument('--config', type=str, default='configs/models/q_obb.yaml', help='Path to the YAML config file')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--save-dir', type=str, default='runs/validate', help='Directory to save validation logs and plots')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset configuration
    with open(args.data, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    active_dataset = dataset_config['active_dataset']
    val_dataset_info = dataset_config['datasets'][active_dataset]['val']

    # Load model from YAML configuration
    model, nc = load_model_from_yaml(args.config)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Move model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # Define transforms and augmentations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        # Add normalization if needed
    ])
    augmentations = None  # Typically, no augmentations during validation

    # Initialize dataloader
    val_dataloader = get_quaternion_dataloader(
        img_dir=val_dataset_info['img_dir'],
        ann_file=val_dataset_info['ann_file'],
        batch_size=4,
        shuffle=False,
        num_workers=10,
        dataset_type=active_dataset.lower(),  # 'dota' or 'coco'
        transform=transform,
        augmentations=augmentations
    )

    # Initialize metrics
    class_names = {0: 'class1', 1: 'class2'}  # Update with actual class names
    metric = OBBMetrics(save_dir=args.save_dir, plot=True, names=class_names)

    # Initialize validator
    validator = Validator(model, val_dataloader, metric, device=device, save_dir=args.save_dir)

    # Perform validation
    print("Starting validation...")
    validator.validate()
    print("Validation completed.")

    # Optionally, plot final metrics
    metric.plot_final_metrics(save_dir=args.save_dir, names=class_names)

    print(f"Validation results and visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()
