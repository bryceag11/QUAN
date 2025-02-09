# train.py
import gc
import os
import torch
import yaml
import argparse
from data.dataloader import get_quaternion_dataloader
from models.model_builder import load_model_from_yaml
from loss.box_loss import DetectionLoss, ClassificationLoss # Changed from BboxLoss
from engine.trainer import Trainer
from utils.metrics import DetMetrics  # Changed from OBBMetrics
from torch.optim import AdamW  # Changed from Adam
from torch.optim.lr_scheduler import OneCycleLR  # Changed from CosineAnnealingLR

def parse_args():
    parser = argparse.ArgumentParser(description="Train Quaternion-based Object Detection Model")
    parser.add_argument('--config', type=str, default='configs/models/q4.yaml', help='Path to model config')
    parser.add_argument('--data', type=str, default='configs/default_config.yaml', help='Path to data config')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')  # Changed from 0.0001
    parser.add_argument('--save-dir', type=str, default='runs/train', help='Save directory')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')  # Added workers arg
    parser.add_argument('--mapping-type', type=str, default='luminance', help='RGB to quaternion mapping type')
    return parser.parse_args()


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def print_model_params(model, indent=''):
    """
    Print parameters for each layer in the model with proper formatting for quaternion layers.
    
    Args:
        model: PyTorch model
        indent: Indentation string for nested layers
    """
    total_params = 0
    trainable_params = 0
    
    # Helper function to format parameter count
    def format_num(num):
        if num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # If it's a leaf module
            num_params = sum(p.numel() for p in module.parameters())
            num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if num_params > 0:  # Only print layers with parameters
                total_params += num_params
                trainable_params += num_trainable
                
                # Special formatting for quaternion layers
                if any(qtype in module.__class__.__name__ for qtype in ['QConv', 'IQBN', 'QBN', 'VQBN']):
                    print(f"{indent}📊 {name} ({module.__class__.__name__})")
                    print(f"{indent}   - Total params: {format_num(num_params)}")
                    print(f"{indent}   - Trainable: {format_num(num_trainable)}")
                    
                    # Print quaternion component info if available
                    if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                        print(f"{indent}   - In channels: {module.in_channels} (quaternion)")
                        print(f"{indent}   - Out channels: {module.out_channels} (quaternion)")
                else:
                    print(f"{indent}🔹 {name} ({module.__class__.__name__})")
                    print(f"{indent}   - Params: {format_num(num_params)}")
    
    # Print total statistics
    print("\n=== Model Summary ===")
    print(f"Total parameters: {format_num(total_params)}")
    print(f"Trainable parameters: {format_num(trainable_params)}")
    print(f"Non-trainable parameters: {format_num(total_params - trainable_params)}")

def main():
    args = parse_args()
    torch.cuda.empty_cache()

    # Set device and CUDA settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load configs
    with open(args.data, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get dataset info
    active_dataset = data_config['active_dataset']
    train_info = data_config['datasets'][active_dataset]['train']
    val_info = data_config['datasets'][active_dataset]['val']
    
    # Create dataloaders
    train_dataloader = get_quaternion_dataloader(
        img_dir=train_info['img_dir'],
        ann_file=train_info['ann_file'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        dataset_type=active_dataset.lower(),
        transform=None,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    val_dataloader = get_quaternion_dataloader(
        img_dir=val_info['img_dir'],
        ann_file=val_info['ann_file'],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        dataset_type=active_dataset.lower(),
        transform=None,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )
    
    # Initialize model
    model, nc = load_model_from_yaml(args.config)
    model = model.to(device)
    print_model_params(model)

    # Count and log parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Mapping type: {args.mapping_type}")
    # Initialize optimizer with better defaults for quaternion networks
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,  # Increased epsilon for stability
        foreach=True,  # Enable more efficient parameter updates
        capturable=True  # Enable capture for CUDA graphs
    )
    
    # Better scheduler for quaternion networks
    scheduler = OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=args.epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,
        div_factor=25,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    # Initialize scaler
    scaler = torch.amp.GradScaler('cuda',
        init_scale=2**10,
        growth_factor=1.50,
        backoff_factor=0.5,
        growth_interval=100
    )
    # Initialize metrics
    metrics = DetMetrics(
        save_dir=args.save_dir,
        plot=True,
        names={i: f'class_{i}' for i in range(nc)}
    )
    
    # Initialize loss function
    loss_fn = ClassificationLoss(nc=80).to(device)
    

    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
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
        grad_clip_val=0.5,
        visualize=True,
        vis_batch_freq=100
    )
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        # Train one epoch
        train_metrics = trainer.train_one_epoch(epoch)
        print(f"\nTraining Results:")
        print(f"Loss: {train_metrics['total_loss']:.4f}")
        print(f"Positive samples: {train_metrics['num_pos']}")
        
        # Validate every 5 epochs and on last epoch
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            print("\nRunning validation...")
            val_metrics = trainer.validate()
            
            # Print validation metrics
            print("\nValidation Results:")
            print(f"Loss: {val_metrics['total_loss']:.4f}")
            print(f"mAP50: {val_metrics['metrics/mAP50(B)']:.4f}")
            print(f"mAP50-95: {val_metrics['metrics/mAP50-95(B)']:.4f}")
            
            # Update and plot metrics
            metrics.update({
                'train/loss': train_metrics['total_loss'],
                'val/loss': val_metrics['total_loss'],
                'val/mAP50': val_metrics['metrics/mAP50(B)'],
                'val/mAP50-95': val_metrics['metrics/mAP50-95(B)']
            })
            
            # Save checkpoint if best validation mAP
            if val_metrics['metrics/mAP50(B)'] > best_map50:
                best_map50 = val_metrics['metrics/mAP50(B)']
                print(f"\nBest mAP50 {best_map50:.4f}")
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'best_map50': best_map50,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, os.path.join(args.save_dir, 'best.pt'))
            
            # Save last checkpoint
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict(),
                    'best_map50': best_map50,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
                torch.save(checkpoint, os.path.join(args.save_dir, f'epoch_{epoch+1}.pt'))
        
        # Plot metrics
        if metrics.curves is not None:
            metrics.plot_curves()
            
        # Update progress
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            print(f"\nLearning rate: {current_lr:.6f}")
            
    print(f"\nTraining completed. Best mAP50: {best_map50:.4f}")

if __name__ == "__main__":
    main()