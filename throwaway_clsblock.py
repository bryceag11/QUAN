


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = IQBN(in_planes)
        self.relu = QPReLU()
        self.conv1 = QConv2D(in_planes, out_planes, kernel_size=1, stride=1,
                           padding=0, bias=False)
        self.droprate = dropRate
        self.pool = QuaternionAvgPool(kernel_size=2, stride=2)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = QuaternionDropout(p=self.droprate)(out)
        return self.pool(out)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class QuaternionDenseNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(QuaternionDenseNet, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        
        # First conv before any dense block - special handling for RGB input
        self.conv1 = QConv2D(3, in_planes, kernel_size=3, stride=1,
                           padding=1, bias=False, mapping_type='raw_normalized')
        
        # Dense blocks
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, 
                                              int(math.floor(in_planes * reduction)), 
                                              dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, 
                                              int(math.floor(in_planes * reduction)), 
                                              dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))
        
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        
        # Global average pooling and classifier
        self.bn1 = IQBN(in_planes)
        self.relu = QPReLU()
        self.fc = QDense(in_planes, num_classes * 4)  # *4 for quaternion output
        self.in_planes = in_planes
        self.global_pool = QuaternionAvgPool()


    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # Use quaternion global pooling
        out = self.global_pool(out)
        out = out.view(-1, self.in_planes)
        out = self.fc(out)
        
        # Extract real components for final output
        batch_size = out.size(0)
        out = out.view(batch_size, -1, 4)  # Reshape to separate quaternion components
        return out[:, :, 0]  # Return real component [batch_size, num_classes]

def create_quaternion_densenet(depth=40, num_classes=10, growth_rate=12, dropRate=0.0):
    """Helper function to create a Quaternion DenseNet with standard configuration"""
    return QuaternionDenseNet(
        depth=depth,
        num_classes=num_classes,
        growth_rate=growth_rate,
        reduction=0.5,
        bottleneck=True,
        dropRate=dropRate
    )



class QuaternionCIFAR10(nn.Module):
    """
    Quaternion CNN model for CIFAR-10 classification.
    """
    def __init__(self, mapping_type='luminance'):
        super(QuaternionCIFAR10, self).__init__()
    
        # PREVIOUSLY WORKING CODE
        # First block
        # Initial convolution block
        self.initial_block = nn.Sequential(
            QConv2D(in_channels=3, out_channels=32, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(32),
            nn.ReLU(),
        )

        # First dense block
        self.block1 = nn.Sequential(
            QConv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(32),
            nn.ReLU(),
            QConv2D(in_channels=32, out_channels=32, kernel_size=3, padding=1, mapping_type=mapping_type),
            # IQBN(32),
            nn.ReLU(),
        )

        # Second convolution block
        self.block2 = nn.Sequential(
            QConv2D(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, mapping_type=mapping_type),
            IQBN(64),
            nn.ReLU(),
            QConv2D(in_channels=64, out_channels=64, kernel_size=3, padding=1, mapping_type=mapping_type),
            # IQBN(64),
            nn.ReLU(),
        )

        # Third convolution block
        self.block3 = nn.Sequential(
            QConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, mapping_type=mapping_type),
            IQBN(128),
            nn.ReLU(),
            QConv2D(in_channels=128, out_channels=128, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(128),
            nn.ReLU(),
        )

        # Fourth convolution block
        self.block4 = nn.Sequential(
            QConv2D(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, mapping_type=mapping_type),
            IQBN(256),
            nn.ReLU(),
            QConv2D(in_channels=256, out_channels=256, kernel_size=3, padding=1, mapping_type=mapping_type),
            IQBN(256),
            nn.ReLU(),
        )

        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.dropout = QuaternionDropout(p=0.1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            QDense(256, 512, mapping_type=mapping_type),
            nn.ReLU(),
            nn.Dropout(0.3),
            QDense(512, NUM_CLASSES * 4, mapping_type=mapping_type)  # Output 4x classes for quaternion
        )
    
    def pool_spatial_only(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies spatial pooling independently for each quaternion component.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, 4, H, W).
            pool_layer (nn.MaxPool2d): Pooling layer to apply.

        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, 4, H_out, W_out).
        """
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components."

        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)

        # Apply pooling
        pooled = self.pool(x_reshaped)

        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

    def avg_pool(self, x: torch.Tensor, num) -> torch.Tensor:
        B, C, Q, H, W = x.shape
        assert Q == 4, "Expected quaternion format with 4 components."

        # Reshape to (B * Q, C, H, W) for spatial pooling
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * Q, C, H, W)

        # Apply pooling
        pooled = F.adaptive_avg_pool2d(x_reshaped, (num, num))

        # Reshape back to (B, C, 4, H_out, W_out)
        H_out, W_out = pooled.shape[-2:]
        return pooled.view(B, Q, C, H_out, W_out).permute(0, 2, 1, 3, 4)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)
        
        # First block with dropout
        x = self.block1(x)
        x = self.pool_spatial_only(x)  # Custom quaternion spatial pooling
        x = self.dropout(x)
        
        # Second block
        x = self.block2(x)
        x = self.pool_spatial_only(x)  # Custom quaternion spatial pooling
        x = self.dropout(x)
        
        # Third block
        x = self.block3(x)
        x = self.pool_spatial_only(x)  # Custom quaternion spatial pooling
        x = self.dropout(x)
        
        # Fourth block
        x = self.block4(x)
        x = self.pool_spatial_only(x)  # Alternate between avg and spatial pooling
        x = self.dropout(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Extract only real components for final classification
        batch_size = x.size(0)
        x = x.view(batch_size, NUM_CLASSES, 4)  # Reshape to separate quaternion components
        real_components = x[:, :, 0]  # Take only real part [batch_size, NUM_CLASSES]
        
        return real_components
