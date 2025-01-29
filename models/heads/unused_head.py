# class QDetectHead(nn.Module):
#     def __init__(self, nc: int, ch: List[int], hidden_dim: int = 256):
#         super().__init__()
#         self.nc = nc
#         self.hidden_dim = hidden_dim - (hidden_dim % 4)
#         self.box_scale = nn.Parameter(torch.tensor(640.0))

#         # Implicit point generators
#         self.point_generators = nn.ModuleList([
#             nn.Sequential(
#                 QConv2D(c, self.hidden_dim, 1),
#                 # QBN(self.hidden_dim // 4),
#                 nn.ReLU(),
#                 QConv2D(self.hidden_dim, 4, 1)  # Generate reference points
#             ) for c in ch
#         ])
        
#         # Adaptive sampling module
#         self.sampling_offsets = nn.ModuleList([
#             nn.Sequential(
#                 QConv2D(c, self.hidden_dim, 1),  # First reduce to hidden_dim from input channels
#                 nn.ReLU(),
#                 QConv2D(self.hidden_dim, 8, 1)  # Then project to 8 channels
#             ) for c in ch
#         ])
        
#         # Detection heads
#         self.det_layers = nn.ModuleList([
#             nn.ModuleDict({
#                 'stem': nn.Sequential(
#                     QConv2D(2*c, self.hidden_dim, 3, padding=1),
#                     # QBN(self.hidden_dim // 4),
#                     nn.ReLU()
#                 ),
#                 'cls': QConv2D(self.hidden_dim, nc, 1),
#                 'reg': QConv2D(self.hidden_dim, 4, 1)
#             }) for c in ch
#         ])

#     def forward(self, features):
#         pred_cls, pred_box = [], []
        
#         for feat, generator, offsets, layer in zip(
#             features, self.point_generators, self.sampling_offsets, self.det_layers):
            
#             # Generate implicit points
#             points = generator(feat).sigmoid() * 640
            
#             # Generate adaptive sampling offsets
#             sample_offsets = offsets(feat).tanh() * 32
            
#             # Sample features at generated points
#             sampled_feats = self._sample_features(feat, points, sample_offsets)
            
#             # Concatenate with original features
#             x = torch.cat([feat, sampled_feats], dim=1)
            
#             # Apply detection heads
#             x = layer['stem'](x)
#             cls_out = layer['cls'](x)
#             box_out = layer['reg'](x).sigmoid() * 640
            
#             pred_cls.append(cls_out)
#             pred_box.append(box_out)
            
#         return pred_cls, pred_box

#     def _sample_features(self, features, points, offsets):
#         """
#         Sample features using learned points and offsets.
#         Implements quaternion-aware deformable sampling.
#         """
#         B, C, Q, H, W = features.shape
#         points = points.sigmoid()  # Normalize to [0,1]
        
#         # Reshape offsets for quaternion structure
#         offsets = offsets.view(B, 2, 4, H, W)
        
#         # Sample features with quaternion preservation
#         sampled = self._quaternion_deform_sample(features, points, offsets)
        
#         return sampled

#     def _quaternion_deform_sample(self, features, points, offsets):
#         """
#         Deformable sampling with quaternion awareness.
        
#         Args:
#             features (torch.Tensor): Input feature map [B, C, 4, H, W]
#             points (torch.Tensor): Sampled points [B, 1, 4, H, W]
#             offsets (torch.Tensor): Sampling offsets [B, 2, 4, H, W]
        
#         Returns:
#             torch.Tensor: Sampled features [B, C, 4, H, W]
#         """
#         B, C, Q, H, W = features.shape
        
#         # Grid generation with learned offsets
#         base_grid_x = torch.linspace(-1, 1, W, device=features.device).view(1, 1, 1, W).expand(B, 1, H, W)
#         base_grid_y = torch.linspace(-1, 1, H, device=features.device).view(1, 1, 1, H).expand(B, 1, H, W)
        
#         # Process offsets for each quaternion component
#         sampled_feats = []
#         for q in range(Q):
#             # Apply offsets to grid for this quaternion component
#             grid_x = base_grid_x + offsets[:, 0, q:q+1] * 2 / W
#             grid_y = base_grid_y + offsets[:, 1, q:q+1] * 2 / H
            
#             # Combine grid
#             grid = torch.cat([grid_x, grid_y], dim=1).permute(0, 2, 3, 1).contiguous()  # [B, H, W, 2]
            
#             # Sample using grid_sample for this quaternion component
#             component_feat = features[:, :, q, :, :]  # [B, C, H, W]
#             sampled_component = F.grid_sample(
#                 component_feat, 
#                 grid, 
#                 mode='bilinear', 
#                 padding_mode='zeros', 
#                 align_corners=False
#             )
#             sampled_feats.append(sampled_component)
        
#         # Stack back into quaternion format
#         sampled = torch.stack(sampled_feats, dim=2)  # [B, C, 4, H, W]
        
#         return sampled


# class QDetectHead(nn.Module):
#     def __init__(self, 
#                  nc: int,  # Number of classes
#                  ch: List[int],  # Input channel sizes for each feature level
#                  hidden_dim: int = 256):
#         super().__init__()
        
#         # Basic configuration
#         self.nc = nc  # Number of classes
#         self.hidden_dim = hidden_dim

#         # Ensure hidden dimension is multiple of 4 for quaternion compatibility
#         self.hidden_dim = self.hidden_dim - (self.hidden_dim % 4)

#         # Detection heads for each feature level
#         self.detection_layers = nn.ModuleList()
#         for channels in ch:
#             # Ensure input channels are multiple of 4
#             assert channels % 4 == 0, f"Input channels must be multiple of 4, got {channels}"
            
#             head = nn.Sequential(
#                 # First quaternion convolution block
#                 QConv2D(channels, self.hidden_dim, kernel_size=3, stride=1, padding=1),
#                 QBN(self.hidden_dim),
#                 QReLU(),
                
#                 # Second quaternion convolution block
#                 QConv2D(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1),
#                 QBN(self.hidden_dim),
#                 QReLU(),
                
#                 # Final conv to get outputs
#                 QConv2D(self.hidden_dim, nc + 4, kernel_size=1)
#             )
#             self.detection_layers.append(head)

#     def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
#         """
#         Forward pass for multi-scale detection
        
#         Args:
#             features (List[torch.Tensor]): Input feature maps from backbone
        
#         Returns:
#             Tuple of class and box predictions
#         """
#         pred_cls, pred_box = [], []
        
#         for feature, layer in zip(features, self.detection_layers):
#             # Ensure feature is in quaternion format [B, C//4, 4, H, W]
#             if feature.dim() == 4:
#                 B, C, H, W = feature.shape
#                 feature = feature.view(B, C//4, 4, H, W)
            
#             # Process through detection head
#             output = layer(feature)
            
#             # Separate class and box predictions
#             B, C, Q, H, W = output.shape
            
#             # Split into class and box predictions
#             cls_pred = output[:, :self.nc, :, :, :]  # Class predictions
#             box_pred = output[:, self.nc:, :, :, :]  # Box predictions
            
#             pred_cls.append(cls_pred)
#             pred_box.append(box_pred)
        
#         return pred_cls, pred_box
