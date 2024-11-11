#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quaternion_layers.conv import QConv2d
from quaternion_layers.dense import QDense

# Configuration
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 250
LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-7
L1_REG = 1e-6
L2_REG = 1e-6
DATA_AUGMENTATION = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'saved_models'
MODEL_NAME = 'pytorch_cifar10_quaternion.pth'

class L1Regularization:
    """L1 regularization for network parameters"""
    def __init__(self, l1_lambda):
        self.l1_lambda = l1_lambda
        
    def __call__(self, model):
        l1_reg = torch.tensor(0., requires_grad=True)
        for name, param in model.named_parameters():
            if 'bias' not in name:
                l1_reg = l1_reg + torch.sum(torch.abs(param))
        return self.l1_lambda * l1_reg

class MetricsLogger:
    """Logger for training and evaluation metrics"""
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.metrics = {
            'train_acc': [],
            'test_acc': [],
            'train_loss': [],
            'test_loss': [],
            'train_reg_loss': [],
            'test_reg_loss': []
        }
    
    def update(self, epoch_metrics):
        """Update metrics with new values"""
        for key, value in epoch_metrics.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save(self, filename='metrics.json'):
        """Save metrics to JSON file"""
        with open(self.save_dir / filename, 'w') as f:
            json.dump(self.metrics, f)
    
    def load(self, filename='metrics.json'):
        """Load metrics from JSON file"""
        with open(self.save_dir / filename, 'r') as f:
            self.metrics = json.load(f)
    
    def plot(self, save_path='training_plots.png'):
        """Create and save visualization plots"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        epochs = range(1, len(self.metrics['train_acc']) + 1)
        
        # Accuracy plot
        ax1.plot(epochs, self.metrics['train_acc'], 'b-', label='Training Accuracy')
        ax1.plot(epochs, self.metrics['test_acc'], 'r-', label='Test Accuracy')
        ax1.set_title('Training and Test Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss')
        ax2.plot(epochs, self.metrics['test_loss'], 'r-', label='Test Loss')
        ax2.plot(epochs, self.metrics['train_reg_loss'], 'g--', label='Train Reg Loss')
        ax2.plot(epochs, self.metrics['test_reg_loss'], 'y--', label='Test Reg Loss')
        ax2.set_title('Training and Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_path)
        plt.close()


class QuaternionCIFAR10(nn.Module):
    """
    Quaternion CNN model for CIFAR-10 classification.
    """
    def __init__(self):
        super(QuaternionCIFAR10, self).__init__()
    
        # PREVIOUSLY WORKING CODE
        # First block
        self.block1 = nn.Sequential(
            QConv2d(in_channels=3, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            QConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Second block
        self.block2 = nn.Sequential(
            QConv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            QConv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(64 * 6 * 6, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1536, NUM_CLASSES)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = F.adaptive_avg_pool2d(x, (6, 6))
        # print(f'After pooling: {x.shape}')  # Should be [batch_size, 64, 6, 6]
        x = self.classifier[0](x)  # Apply Flatten
        # print(f'After flattening: {x.shape}')  # Should be [batch_size, 2304]
        x = self.classifier[1](x)  # Apply QDense
        # print(f'After QDense: {x.shape}')  # Should be [batch_size, 1536]
        x = self.classifier[2](x)  # Apply ReLU
        x = self.classifier[3](x)  # Apply Dropout
        x = self.classifier[4](x)  # Apply Linear
        return x


def get_data_loaders():
    """
    Prepare CIFAR-10 data loaders with augmentation.
    """
    if DATA_AUGMENTATION:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return trainloader, testloader

def train_epoch(model, train_loader, criterion, optimizer, l1_reg, epoch, writer):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Calculate losses
        criterion_loss = criterion(outputs, targets)
        l1_loss = l1_reg(model)
        total_loss = criterion_loss + l1_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += criterion_loss.item()
        running_reg_loss += l1_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 99:
            writer.add_scalar('training loss', 
                            running_loss / 100, 
                            epoch * len(train_loader) + batch_idx)
            writer.add_scalar('l1 regularization loss',
                            running_reg_loss / 100,
                            epoch * len(train_loader) + batch_idx)
            running_loss = 0.0
            running_reg_loss = 0.0
    
    acc = 100. * correct / total
    writer.add_scalar('training accuracy', acc, epoch)
    avg_loss = running_loss / len(train_loader)
    avg_reg_loss = running_reg_loss / len(train_loader)
    return acc, avg_loss, avg_reg_loss    

def evaluate(model, test_loader, criterion, l1_reg, epoch, writer):
    """
    Evaluate the model on test set.
    """
    model.eval()
    test_loss = 0
    test_reg_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            
            criterion_loss = criterion(outputs, targets)
            l1_loss = l1_reg(model)
            total_loss = criterion_loss + l1_loss
            
            test_loss += criterion_loss.item()
            test_reg_loss += l1_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100. * correct / total
    writer.add_scalar('test accuracy', acc, epoch)
    writer.add_scalar('test loss', test_loss / len(test_loader), epoch)
    writer.add_scalar('test reg loss', test_reg_loss / len(test_loader), epoch)
    avg_loss = test_loss / len(test_loader)
    avg_reg_loss = test_reg_loss / len(test_loader)
    return acc, avg_loss, avg_reg_loss


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    writer = SummaryWriter('runs/cifar10_quaternion_adam')
    metrics_logger = MetricsLogger(SAVE_DIR)
    
    train_loader, test_loader = get_data_loaders()
    
    model = QuaternionCIFAR10().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA_1, BETA_2),
        eps=EPSILON,
        weight_decay=L2_REG
    )
    l1_reg = L1Regularization(L1_REG)
    
    print('Starting training...')
    print(f'Training on device: {DEVICE}')
    print(f'Adam parameters: lr={LEARNING_RATE}, beta1={BETA_1}, beta2={BETA_2}')
    print(f'Regularization: L1={L1_REG}, L2={L2_REG}')
    
    best_acc = 0
    epoch_metrics = {
        'train_acc': [],
        'test_acc': [],
        'train_loss': [],
        'test_loss': [],
        'train_reg_loss': [],
        'test_reg_loss': []
    }
    
    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch+1}/{EPOCHS}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_reg_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate losses
            criterion_loss = criterion(outputs, targets)
            l1_loss = l1_reg(model)
            total_loss = criterion_loss + l1_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Track metrics
            running_loss += criterion_loss.item()
            running_reg_loss += l1_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 99:
                writer.add_scalar('training loss', 
                                running_loss / 100, 
                                epoch * len(train_loader) + batch_idx)
                writer.add_scalar('l1 regularization loss',
                                running_reg_loss / 100,
                                epoch * len(train_loader) + batch_idx)
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {running_loss/100:.3f} '
                      f'Reg Loss: {running_reg_loss/100:.3f}')
                running_loss = 0.0
                running_reg_loss = 0.0
        
        train_acc = 100. * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_train_reg_loss = running_reg_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        test_reg_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                
                criterion_loss = criterion(outputs, targets)
                l1_loss = l1_reg(model)
                
                test_loss += criterion_loss.item()
                test_reg_loss += l1_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_acc = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        avg_test_reg_loss = test_reg_loss / len(test_loader)
        
        # Log metrics
        metrics_logger.update({
            'train_acc': train_acc,
            'test_acc': test_acc,
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss,
            'train_reg_loss': avg_train_reg_loss,
            'test_reg_loss': avg_test_reg_loss
        })
        
        # Save metrics
        metrics_logger.save()
        
        # Log to tensorboard
        writer.add_scalar('training accuracy', train_acc, epoch)
        writer.add_scalar('test accuracy', test_acc, epoch)
        writer.add_scalar('test loss', avg_test_loss, epoch)
        writer.add_scalar('test reg loss', avg_test_reg_loss, epoch)
        
        # Print epoch summary
        print(f'Epoch {epoch + 1}/{EPOCHS}:')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Test Loss: {avg_test_loss:.4f}')
        
        # Create visualization every 10 epochs
        if (epoch + 1) % 10 == 0:
            metrics_logger.plot()
        
        # Save best model
        if test_acc > best_acc:
            print('Saving model...')
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'hyperparameters': {
                    'learning_rate': LEARNING_RATE,
                    'beta1': BETA_1,
                    'beta2': BETA_2,
                    'epsilon': EPSILON,
                    'l1_reg': L1_REG,
                    'l2_reg': L2_REG,
                }
            }, os.path.join(SAVE_DIR, MODEL_NAME))
    
    # Final visualization
    metrics_logger.plot('final_training_plots.png')
    
    writer.close()
    print(f'Best test accuracy: {best_acc:.2f}%')
    print('Finished Training')
    
    # Save final metrics summary
    summary = {
        'best_accuracy': best_acc,
        'final_train_acc': train_acc,
        'final_test_acc': test_acc,
        'final_train_loss': avg_train_loss,
        'final_test_loss': avg_test_loss,
    }
    with open(os.path.join(SAVE_DIR, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == '__main__':
    main()