import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResolutionCNN_shallow(nn.Module):
    """
    Simplified Shallow CNN for RGB image super-resolution: 64x64 -> 256x256 (4x upsampling)
    Removed identity convolution layers to reduce parameters
    Direct path: Input -> Downsample -> Upsample stages -> Output
    Input: 3 channels (RGB), Output: 3 channels (RGB)
    """
    def __init__(self, num_channels=3):
        super(SuperResolutionCNN_shallow, self).__init__()
        
        # Direct downsampling: 64x64 -> 32x32 (32 features)
        self.downsample = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn_down = nn.BatchNorm2d(32)
        
        # First upsampling: 32x32 -> 64x64 (64 features)
        self.upsample1 = nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up1 = nn.BatchNorm2d(64)
        
        # Second upsampling: 64x64 -> 128x128 (128 features)
        self.upsample2 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up2 = nn.BatchNorm2d(128)
        
        # Final upsampling: 128x128 -> 256x256 (64 features, then reduce to output)
        self.upsample3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up3 = nn.BatchNorm2d(64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.skip_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Store input for skip connection
        skip = self.skip_upsample(x)
        
        # Simplified encoder-decoder path
        # Downsample: 64x64 -> 32x32 (32 features)
        x = F.relu(self.bn_down(self.downsample(x)))
        
        # Progressive upsampling with feature expansion
        # 32x32 -> 64x64 (64 features)
        x = F.relu(self.bn_up1(self.upsample1(x)))
        
        # 64x64 -> 128x128 (128 features) 
        x = F.relu(self.bn_up2(self.upsample2(x)))
        
        # 128x128 -> 256x256 (64 features)
        x = F.relu(self.bn_up3(self.upsample3(x)))
        
        # Final output: 256x256 (3 channels)
        x = torch.sigmoid(self.final_conv(x))
        
        # Add skip connection
        return x + skip
    

class SuperResolutionCNN_deep(nn.Module):
    """
    Deep CNN for RGB image super-resolution: 64x64 -> 256x256 (4x upsampling)
    Architecture: 64-32-64-128-256 feature channels with deeper layers
    Input: 3 channels (RGB), Output: 3 channels (RGB)
    """
    def __init__(self, num_channels=3):
        super(SuperResolutionCNN_deep, self).__init__()
        
        # Encoder following 64-32-64 pattern
        # Initial: 64x64 -> 64x64 (64 features)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Downsample: 64x64 -> 32x32 (32 features)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Deep feature extraction at bottleneck: 32x32 (64 features)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Bottleneck with residual connections
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Decoder following 128-128-256 pattern
        # Upsample: 32x32 -> 64x64 (128 features)
        self.deconv1 = nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        # Upsample: 64x64 -> 128x128 (128 features)
        self.deconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        
        # Final upsample: 128x128 -> 256x256 (256 features)
        self.deconv3 = nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        
        # Refinement layers
        self.refine1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.refine2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(64)
        
        # Final output
        self.final_conv = nn.Conv2d(64, num_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.skip_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Store input for skip connection
        skip = self.skip_upsample(x)
        
        # Encoder: 64-32-64 architecture
        x = F.relu(self.bn1(self.conv1(x)))        # 64 features at 64x64
        x = F.relu(self.bn2(self.conv2(x)))        # 32 features at 32x32
        x = F.relu(self.bn3(self.conv3(x)))        # 64 features at 32x32
        x = F.relu(self.bn4(self.conv4(x)))        # 64 features at 32x32

        # Bottleneck with residual
        identity = x
        x = self.bottleneck(x)
        x = x + identity  # Residual connection

        # Decoder: 128-128-256 architecture
        x = F.relu(self.bn5(self.deconv1(x)))      # 128 features at 64x64
        x = F.relu(self.bn6(self.deconv2(x)))      # 128 features at 128x128
        x = F.relu(self.bn7(self.deconv3(x)))      # 256 features at 256x256
        
        # Refinement
        x = F.relu(self.bn8(self.refine1(x)))      # 128 features at 256x256
        x = F.relu(self.bn9(self.refine2(x)))      # 64 features at 256x256
        x = torch.sigmoid(self.final_conv(x))      # 3 channels at 256x256
        
        # Add skip connection
        return x + skip
    
    
class UNet(nn.Module):
    """
    UNet for RGB image super-resolution: 64x64 -> 256x256 (4x upsampling)
    Architecture: 64-32-64-128-256 feature channels with skip connections
    Input: 3 channels (RGB), Output: 3 channels (RGB)
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 32, 64]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder following 64-32-64 pattern
        prev_channels = in_channels
        for feature in features:
            self.encoder_layers.append(self.double_conv(prev_channels, feature))
            prev_channels = feature

        # Bottleneck at 32x32 with 64 features
        self.bottleneck = self.double_conv(features[-1], features[-1])

        # Decoder following 128-128-256 pattern
        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        
        # First upsampling: 32x32 -> 64x64 (128 features)
        self.upconvs.append(nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2))
        self.decoder_layers.append(self.double_conv(128 + 32, 128))  # 128 + skip connection from encoder
        
        # Second upsampling: 64x64 -> 128x128 (128 features) 
        self.upconvs.append(nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2))
        self.decoder_layers.append(self.double_conv(128 + 64, 128))  # 128 + skip connection
        
        # Final upsampling: 64x64 -> 256x256 (256 features then reduce to output channels)
        self.final_up = nn.ConvTranspose2d(128, 256, kernel_size=4, stride=4)
        self.final_refine = self.double_conv(256, 128)
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # Skip connection for residual learning
        self.skip_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Store input for skip connection
        skip_input = self.skip_upsample(x)
        
        # Encoder with skip connections (64-32-64 features)
        encs = []
        for enc in self.encoder_layers:
            x = enc(x)
            encs.append(x)
            if len(encs) < len(self.encoder_layers):  # Don't pool after last encoder layer
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections (128-128-256 features)
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = encs[-(idx+1)]  # Get corresponding encoder features
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_layers[idx](x)
        
        # Final upsampling to 256x256 with 256 features
        x = self.final_up(x)
        x = self.final_refine(x)
        x = torch.sigmoid(self.final_conv(x))
        
        # Add skip connection
        return x + skip_input
    
    
class SRCNN_4x(nn.Module):
    """
    SRCNN adapted for RGB image super-resolution: 64x64 -> 256x256 (4x upsampling)
    Architecture: 64-32-64-128-256 feature channels using pixel shuffle upsampling
    Input: 3 channels (RGB), Output: 3 channels (RGB)
    """
    
    def __init__(self, num_channels=3):
        super(SRCNN_4x, self).__init__()
        
        # Feature extraction following 64-32-64 pattern
        self.feature_extraction = nn.Sequential(
            # Initial: 64x64 -> 64x64 (64 features)
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Reduce to 32 features
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Expand to 64 features
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Non-linear mapping - enhance to 128 features
        self.mapping = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling branch - 2x upsampling to 128x128
        self.upsample_2x = nn.Sequential(
            nn.Conv2d(128, 128 * 4, kernel_size=3, padding=1),  # 4 = 2x2 upsampling
            nn.PixelShuffle(2),  # 128x128 with 128 features
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Another 2x upsampling to 256x256 with 256 features
        self.upsample_4x = nn.Sequential(
            nn.Conv2d(128, 256 * 4, kernel_size=3, padding=1),  # 4 = 2x2 upsampling  
            nn.PixelShuffle(2),  # 256x256 with 256 features
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Reconstruction - reduce from 256 features to output channels
        self.reconstruction = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # RGB output in [0,1] range
        )
        
        # Skip connection
        self.skip_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Store input for skip connection
        skip = self.skip_upsample(x)
        
        # Feature extraction (64-32-64 features)
        features = self.feature_extraction(x)  # 64x64 with 64 features
        
        # Non-linear mapping (128 features)
        mapped = self.mapping(features)  # 64x64 with 128 features
        
        # Progressive upsampling (128-256 features)
        upsampled_2x = self.upsample_2x(mapped)  # 128x128 with 128 features
        upsampled_4x = self.upsample_4x(upsampled_2x)  # 256x256 with 256 features
        
        # Final reconstruction
        output = self.reconstruction(upsampled_4x)  # 256x256 with 3 channels
        
        # Add skip connection
        return output + skip


class EnhancedSRCNN_4x(nn.Module):
    """
    Enhanced SRCNN for RGB image super-resolution: 64x64 -> 256x256 (4x upsampling)
    Architecture: 64-32-64-128-256 feature channels as requested
    Input: 3 channels (RGB), Output: 3 channels (RGB)
    """
    
    def __init__(self, num_channels=3):
        super(EnhancedSRCNN_4x, self).__init__()
        
        # Initial feature extraction: 64x64 -> 64x64 (64 features)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Downsampling: 64x64 -> 32x32 (32 features) 
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Feature extraction at bottleneck: 32x32 -> 32x32 (64 features)
        self.feature_extract = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling back to 64x64 (128 features)
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Progressive upsampling to 128x128 (128 features)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling to 256x256 (256 features)
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Final reconstruction layers
        self.reconstruction = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output to [0,1] for RGB values
        )
        
        # Skip connection upsampling (4x upsampling for residual connection)
        self.skip_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Store input for skip connection (4x upsampling)
        input_skip = self.skip_upsample(x)
        
        # Feature extraction: 64x64 with 64 features
        x = self.initial_conv(x)
        
        # Downsample: 64x64 -> 32x32 with 32 features
        x = self.downsample(x)
        
        # Feature extraction at bottleneck: 32x32 with 64 features
        x = self.feature_extract(x)
        
        # Progressive upsampling following 64-32-64-128-256 architecture
        x = self.upsample1(x)     # 32x32 -> 64x64, 128 features
        x = self.upsample2(x)     # 64x64 -> 128x128, 128 features  
        x = self.upsample3(x)     # 128x128 -> 256x256, 256 features
        
        # Final reconstruction
        x = self.reconstruction(x)  # 256x256 -> 256x256, 3 channels
        
        # Add skip connection (residual learning)
        x = x + input_skip
        
        return x
    

def edge_preserving_loss(pred, target):
    """
    Compute edge-preserving loss using Sobel operators for RGB images.
    
    Args:
        pred: model predictions (B, C, H, W)
        target: ground truth (B, C, H, W)
        
    Returns:
        edge_loss: edge-preserving loss
    """
    # Sobel operators for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    
    # Expand Sobel operators for each channel
    num_channels = pred.shape[1]
    sobel_x = sobel_x.repeat(num_channels, 1, 1, 1)
    sobel_y = sobel_y.repeat(num_channels, 1, 1, 1)
    
    # Apply Sobel operators to each channel separately
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=num_channels)
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=num_channels)
    target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=num_channels)
    target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=num_channels)
    
    edge_loss = F.mse_loss(pred_edge_x, target_edge_x) + F.mse_loss(pred_edge_y, target_edge_y)
    return edge_loss

def gradient_loss(pred, target):
    """
    Compute gradient loss for RGB images by comparing spatial gradients.
    
    Args:
        pred: model predictions (B, C, H, W)
        target: ground truth (B, C, H, W)
        
    Returns:
        grad_loss: gradient loss
    """
    # Compute gradients in x and y directions for all channels
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    return grad_loss_x + grad_loss_y