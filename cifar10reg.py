class BasicBlock(nn.Module):
    """Standard ResNet basic block"""
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                            stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU()  # Same activation as in quaternion implementation
        self.dropout1 = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Second convolution block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # Shortcut connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        # First block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second block
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Add residual connection
        out += identity
        out = self.relu(out)
        
        return out

class ResNet34(nn.Module):
    """
    Standard ResNet34 implementation matching the quaternion version's structure
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.dropout_rates = {
            'initial': [.1, .15, .2, .2, .25],  # [block2, block3, block4, block5, classifier]
            'increment': 0.05  # Amount to increase after each LR drop
        }
        self.current_rates = self.dropout_rates['initial'].copy()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        
        # ResNet blocks
        self.conv2_x = self._make_layer(64, 64, 3, 1, dropout_idx=0)
        self.conv3_x = self._make_layer(64, 128, 4, 2, dropout_idx=1)
        self.conv4_x = self._make_layer(128, 256, 6, 2, dropout_idx=2)
        self.conv5_x = self._make_layer(256, 512, 3, 2, dropout_idx=3)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=self.current_rates[4]),
            nn.Linear(512, num_classes)
        )

    def update_dropout_rates(self):
        """Increase dropout rates by the increment amount"""
        for i in range(len(self.current_rates)):
            self.current_rates[i] = min(0.5, self.current_rates[i] + self.dropout_rates['increment'])
            
        # Update dropout in all blocks
        self._update_block_dropout(self.conv2_x, 0)
        self._update_block_dropout(self.conv3_x, 1)
        self._update_block_dropout(self.conv4_x, 2)
        self._update_block_dropout(self.conv5_x, 3)
        
        # Update classifier dropout
        if isinstance(self.classifier[3], nn.Dropout):
            self.classifier[3].p = self.current_rates[4]

    def _update_block_dropout(self, block, rate_idx):
        """Update dropout rates in a block"""
        for layer in block:
            if isinstance(layer, BasicBlock):
                if isinstance(layer.dropout1, nn.Dropout):
                    layer.dropout1.p = self.current_rates[rate_idx]
                if isinstance(layer.dropout2, nn.Dropout):
                    layer.dropout2.p = self.current_rates[rate_idx]

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, dropout_idx):
        """Create a layer of residual blocks with dynamic dropout rates"""
        layers = []
        
        # First block handles stride and channel changes
        layers.append(BasicBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            dropout_rate=self.current_rates[dropout_idx]
        ))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                dropout_rate=self.current_rates[dropout_idx]
            ))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial conv
        x = self.conv1(x)
        
        # Residual blocks
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        
        # Global average pooling
        x = self.gap(x)
        
        # Classifier
        x = self.classifier(x)
        
        return x


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create model
    model = ResNet34(num_classes=10)
    
    # Print model parameter count
    num_params = count_parameters(model)
    print(f'Total trainable parameters: {num_params:,}')