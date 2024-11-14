# train.py
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import torch
import yaml
import argparse
from data.dataloader import get_quaternion_dataloader
from models.model_builder import load_model_from_yaml
from loss.box_loss import DetectionLoss, BboxLoss  # Import DetectionLoss instead of BboxLoss and RotatedBBoxLoss
from engine.trainer import Trainer
from utils.metrics import OBBMetrics, DetMetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
import torch.backends.cudnn as cudnn

def parse_args():
    parser = argparse.ArgumentParser(description="Train Quaternion-based Object Detection Model")
    parser.add_argument('--config', type=str, default='configs/models/q.yaml', help='Path to model config')
    parser.add_argument('--data', type=str, required=True, help='Path to data config')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Save directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    return parser.parse_args()

def main():
    
    # Parse arguments
    args = parse_args()
    

    torch.cuda.empty_cache()  # Clear cache at start of epoch

    # Set device and CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset configuration
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get dataset info
    active_dataset = data_config['active_dataset']
    train_info = data_config['datasets'][active_dataset]['train']
    val_info = data_config['datasets'][active_dataset]['val']
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataloaders
    train_dataloader = get_quaternion_dataloader(
        img_dir=train_info['img_dir'],
        ann_file=train_info['ann_file'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        dataset_type=active_dataset.lower(),
        transform=transform,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2
    )
    
    val_dataloader = get_quaternion_dataloader(
        img_dir=val_info['img_dir'],
        ann_file=val_info['ann_file'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        dataset_type=active_dataset.lower(),
        transform=transform,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2
    )
    
    # Initialize model
    model, nc = load_model_from_yaml(args.config)
    model = model.to(device)
    
    # Check if model has parameters
    if not any(p.requires_grad for p in model.parameters()):
        raise ValueError("The loaded model has no trainable parameters. Please check the model configuration.")
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
        amsgrad=False  # Disable AMS-Grad to save memory
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize metrics
    metrics = DetMetrics(
        save_dir=args.save_dir,
        plot=True,
        names={i: f'class_{i}' for i in range(nc)}  # Or your actual class names
    )
    
    # Choose appropriate loss function based on dataset type
    if active_dataset.lower() == 'dota':
        loss_fn = DetectionLoss(reg_max=16, nc=nc, use_quat=True, tal_topk=10)  # For oriented bounding boxes
    else:
        loss_fn = BboxLoss(reg_max=16, nc=80, use_quat=False)  # For axis-aligned bounding boxes
    
    # Move loss function to device if necessary
    loss_fn = loss_fn.to(device)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'Resumed from epoch {start_epoch}')
        else:
            print(f"No checkpoint found at '{args.resume}'. Starting from scratch.")
    

        # Initialize mixed precision scaler
    scaler = torch.amp.GradScaler(
        init_scale=2**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000
    )
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_fn=loss_fn,
        metrics=metrics,
        device=device,
        save_dir=args.save_dir,
        grad_clip_val=1.0,
        visualize=True,
        vis_batch_freq=100
    )
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Clear cache before each epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Train and validate
        train_metrics = trainer.train_one_epoch(epoch)
        
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            torch.cuda.empty_cache()
            val_metrics = trainer.validate()
            print("\nValidation Results:")
            for k, v in val_metrics.items():
                print(f"{k}: {v:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(epoch)
        
        # Update and plot metrics
        if 'val_metrics' in locals():
            metrics.update(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics)
            metrics.plot()

if __name__ == "__main__":
    main()
