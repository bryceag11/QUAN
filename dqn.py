import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from tqdm import tqdm

# Import our quaternion modules
from quat import (
    QuaternionConv2D, 
    QuaternionBatchNorm, 
    GetR, GetI, GetJ, GetK,
    QuaternionDense
)

class QuaternionResidualBlock(nn.Module):
    """
    Quaternion Residual Block for Deep Quaternion Networks
    
    Args:
        in_channels: Number of quaternion input channels (total channels / 4)
        out_channels: Number of quaternion output channels
        stride: Stride for the first convolution in the block
        shortcut: Type of shortcut connection ('regular' or 'projection')
    """
    def __init__(self, in_channels, out_channels, stride=1, shortcut='regular'):
        super(QuaternionResidualBlock, self).__init__()
        self.shortcut = shortcut
        
        # First conv with normalization and activation
        self.bn1 = QuaternionBatchNorm(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = QuaternionConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        
        # Second conv with normalization and activation
        self.bn2 = QuaternionBatchNorm(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = QuaternionConv2D(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # Shortcut connection
        if shortcut == 'projection' or in_channels != out_channels:
            if shortcut == 'projection':
                self.shortcut_conv = QuaternionConv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                )
            else:
                self.shortcut_conv = QuaternionConv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False
                )
        else:
            self.shortcut_conv = None
    
    def forward(self, x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        if self.shortcut_conv is not None:
            identity = self.shortcut_conv(x)
            
        out += identity
        return out


class QuaternionVectorBlock(nn.Module):
    """
    Block to learn imaginary components from real input
    
    Args:
        in_channels: Number of real input channels
        filter_size: Size of convolution kernel
    """
    def __init__(self, in_channels, filter_size=3):
        super(QuaternionVectorBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=filter_size,
            padding=filter_size//2,
            bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=filter_size,
            padding=filter_size//2,
            bias=False
        )
    
    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        
        return out


class QuaternionResNet(nn.Module):
    """
    Quaternion Residual Network
    
    Args:
        num_blocks: List with number of residual blocks in each stage
        start_filters: Number of filters in the first stage
        num_classes: Number of output classes
        input_channels: Number of input channels (usually 3 for RGB)
    """
    def __init__(
        self, 
        num_blocks=[2, 1, 1], 
        start_filters=8, 
        num_classes=10, 
        input_channels=3
    ):
        super(QuaternionResNet, self).__init__()
        self.in_channels = input_channels
        self.quat_channels = start_filters
        
        # Learn the quaternion components from real input
        self.vector_i = QuaternionVectorBlock(input_channels)
        self.vector_j = QuaternionVectorBlock(input_channels)
        self.vector_k = QuaternionVectorBlock(input_channels)
        
        # Initial convolution
        self.conv1 = QuaternionConv2D(
            in_channels=input_channels,
            out_channels=self.quat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # Create the stages with residual blocks
        self.stages = nn.ModuleList()
        in_channels = self.quat_channels
        
        # First stage
        layers = []
        for i in range(num_blocks[0]):
            layers.append(QuaternionResidualBlock(in_channels, self.quat_channels))
            in_channels = self.quat_channels
        self.stages.append(nn.Sequential(*layers))
        
        # Subsequent stages
        for stage_idx in range(1, len(num_blocks)):
            layers = []
            
            # First block uses projection shortcut to downsample
            layers.append(QuaternionResidualBlock(
                in_channels, 
                self.quat_channels * (2 ** stage_idx),
                stride=2,
                shortcut='projection'
            ))
            
            # Remaining blocks in the stage
            in_channels = self.quat_channels * (2 ** stage_idx)
            for i in range(1, num_blocks[stage_idx]):
                layers.append(QuaternionResidualBlock(in_channels, in_channels))
            
            self.stages.append(nn.Sequential(*layers))
        
        # Final norm, pooling and classification layers
        final_channels = self.quat_channels * (2 ** (len(num_blocks) - 1))
        self.bn = QuaternionBatchNorm(final_channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_channels * 4, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Create quaternion input from real input
        real = x
        imag_i = self.vector_i(x)
        imag_j = self.vector_j(x)
        imag_k = self.vector_k(x)
        
        # Concatenate quaternion components
        x = torch.cat([real, imag_i, imag_j, imag_k], dim=1)
        
        # Forward pass through the network
        x = self.conv1(x)
        
        # Pass through each stage
        for stage in self.stages:
            x = stage(x)
        
        # Final layers
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


# Create models for comparison
def create_real_resnet(num_blocks, start_filters, num_classes=10):
    """Create a standard real-valued ResNet"""
    
    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1, shortcut='regular'):
            super(ResidualBlock, self).__init__()
            self.shortcut = shortcut
            
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                  stride=stride, padding=1, bias=False)
            
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                  stride=1, padding=1, bias=False)
            
            if shortcut == 'projection' or in_channels != out_channels:
                if shortcut == 'projection':
                    self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                 stride=stride, bias=False)
                else:
                    self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                 stride=1, bias=False)
            else:
                self.shortcut_conv = None
        
        def forward(self, x):
            identity = x
            
            out = self.bn1(x)
            out = self.relu1(out)
            out = self.conv1(out)
            
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.conv2(out)
            
            if self.shortcut_conv is not None:
                identity = self.shortcut_conv(x)
                
            out += identity
            return out
    
    class ResNet(nn.Module):
        def __init__(self, num_blocks, start_filters, num_classes=10, input_channels=3):
            super(ResNet, self).__init__()
            self.in_channels = input_channels
            self.channels = start_filters
            
            # Initial convolution
            self.conv1 = nn.Conv2d(input_channels, self.channels, kernel_size=3,
                                 stride=1, padding=1, bias=False)
            
            # Create stages
            self.stages = nn.ModuleList()
            in_channels = self.channels
            
            # First stage
            layers = []
            for i in range(num_blocks[0]):
                layers.append(ResidualBlock(in_channels, self.channels))
                in_channels = self.channels
            self.stages.append(nn.Sequential(*layers))
            
            # Subsequent stages
            for stage_idx in range(1, len(num_blocks)):
                layers = []
                
                layers.append(ResidualBlock(
                    in_channels, 
                    self.channels * (2 ** stage_idx),
                    stride=2,
                    shortcut='projection'
                ))
                
                in_channels = self.channels * (2 ** stage_idx)
                for i in range(1, num_blocks[stage_idx]):
                    layers.append(ResidualBlock(in_channels, in_channels))
                
                self.stages.append(nn.Sequential(*layers))
            
            # Final layers
            final_channels = self.channels * (2 ** (len(num_blocks) - 1))
            self.bn = nn.BatchNorm2d(final_channels)
            self.relu = nn.ReLU(inplace=True)
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(final_channels, num_classes)
            
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.conv1(x)
            
            for stage in self.stages:
                x = stage(x)
            
            x = self.bn(x)
            x = self.relu(x)
            x = self.avg_pool(x)
            x = self.flatten(x)
            x = self.fc(x)
            
            return x
    
    return ResNet(num_blocks, start_filters, num_classes)


# Training function
def train_model(model, train_loader, test_loader, epochs=200, device='cuda', 
                clipnorm=1.0, filename=None, dataset='cifar10'):
    """
    Train and evaluate a model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of epochs to train
        device: Device to use for training ('cuda' or 'cpu')
        clipnorm: Maximum gradient norm for clipping
        filename: Filename to save the model weights
        dataset: Dataset name ('cifar10' or 'cifar100')
    
    Returns:
        Accuracy on test set
    """
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Learning rate scheduler
    def lr_schedule(epoch):
        if epoch < 10:
            return 0.01
        elif epoch < 100:
            return 0.1
        elif epoch < 120:
            return 0.1
        elif epoch < 150:
            return 0.01
        else:
            return 0.001
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    # Track metrics
    best_acc = 0.0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        print(f"Epoch {epoch+1}/{epochs}, LR: {optimizer.param_groups[0]['lr']}")
        
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
            
            # Update weights
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Testing
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Save best model
        if test_acc > best_acc and filename is not None:
            best_acc = test_acc
            torch.save(model.state_dict(), f"{filename}_weights.pth")
        
        # Update learning rate
        scheduler.step()
    
    # Save metrics
    if filename is not None:
        np.savetxt(f"{filename}_train_loss.txt", np.array(train_losses))
        np.savetxt(f"{filename}_train_acc.txt", np.array(train_accs))
        np.savetxt(f"{filename}_test_loss.txt", np.array(test_losses))
        np.savetxt(f"{filename}_test_acc.txt", np.array(test_accs))
    
    return best_acc


def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Setup data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # Create data loaders
    trainloader = DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2)
    
    # Create models
    shallow_real = create_real_resnet([2, 1, 1], 32, num_classes=10)
    shallow_quaternion = QuaternionResNet([2, 1, 1], 12, num_classes=10)
    
    # Print model sizes
    real_params = sum(p.numel() for p in shallow_real.parameters())
    quat_params = sum(p.numel() for p in shallow_quaternion.parameters())
    
    print(f"Real Model Parameters: {real_params}")
    print(f"Quaternion Model Parameters: {quat_params}")
    
    # Train models
    # Train quaternion model
    print("\nTraining Quaternion Model:")
    train_model(shallow_quaternion, trainloader, testloader, epochs=200, 
                device=device, filename="quaternion", dataset='cifar10')
    
    # Load CIFAR-100 dataset for second experiment
    # trainset_100 = torchvision.datasets.CIFAR100(
    #     root='./data', train=True, download=True, transform=transform_train)
    # testset_100 = torchvision.datasets.CIFAR100(
    #     root='./data', train=False, download=True, transform=transform_test)
    
    # Create CIFAR-100 data loaders
    # trainloader_100 = DataLoader(
    #     trainset_100, batch_size=128, shuffle=True, num_workers=2)
    # testloader_100 = DataLoader(
    #     testset_100, batch_size=100, shuffle=False, num_workers=2)
    
    # Create CIFAR-100 models
    # shallow_quaternion_100 = QuaternionResNet([2, 1, 1], 8, num_classes=100)
    

    # print("\nTraining Quaternion Model (CIFAR-100):")
    # train_model(shallow_quaternion_100, trainloader_100, testloader_100, epochs=200, 
    #             device=device, filename="quaternion_cifar100", dataset='cifar100')
    
    # Create deep models for both CIFAR-10 and CIFAR-100
    deep_quaternion = QuaternionResNet([10, 9, 9], 8, num_classes=10)
    
    deep_quaternion_100 = QuaternionResNet([10, 9, 9], 8, num_classes=100)
    
    # Print deep model sizes
    deep_quat_params = sum(p.numel() for p in deep_quaternion.parameters())
    
    print(f"Deep Quaternion Model Parameters: {deep_quat_params}")
    

    
    print("\nTraining Deep Quaternion Model (CIFAR-10):")
    # train_model(deep_quaternion, trainloader, testloader, epochs=200, 
    #             device=device, filename="deep_quaternion", dataset='cifar10')
    
    
    # print("\nTraining Deep Quaternion Model (CIFAR-100):")
    # train_model(deep_quaternion_100, trainloader_100, testloader_100, epochs=200, 
    #             device=device, filename="deep_quaternion_cifar100", dataset='cifar100')
    
    print("\nExperiment completed. Summary of model parameters:")
    print(f"Shallow Quaternion Model: {quat_params}")
    # print(f"Deep Quaternion Model: {deep_quat_params}")


if __name__ == "__main__":
    main()