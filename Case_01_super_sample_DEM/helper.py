import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResolutionCNN_shallow(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN_shallow, self).__init__()
        
        # Encoder: 128x128 -> 64x64
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Further encode: 64x64 -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Start decoding: 32x32 -> 64x64
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Decode: 64x64 -> 128x128
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Super-resolution: 128x128 -> 384x384 (3x upscaling)
        # Edge effects will be handled during inference with overlapping tiles
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=3, padding=1, output_padding=2)
        
    def forward(self, x):
        # Encoder: 128 -> 64 -> 32
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Decoder: 32 -> 64 -> 128 -> 384
        # Edge effects will be handled during inference with overlapping tiles
        x = F.relu(self.bn3(self.deconv1(x)))
        x = F.relu(self.bn4(self.deconv2(x)))
        x = self.deconv3(x)  # Outputs 384x384 tensor
        
        return x
    

class SuperResolutionCNN_deep(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN_deep, self).__init__()
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # Decoder: 16x16 -> 32x32 -> 64x64 -> 128x128 -> 384x384
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=3, padding=1, output_padding=2)
        self.bn9 = nn.BatchNorm2d(16)
        self.final_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = F.relu(self.bn5(self.deconv1(x)))
        x = F.relu(self.bn6(self.deconv2(x)))
        x = F.relu(self.bn7(self.deconv3(x)))
        x = F.relu(self.bn8(self.deconv4(x)))
        x = F.relu(self.bn9(self.deconv5(x)))
        x = self.final_conv(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        prev_channels = in_channels
        for feature in features:
            self.encoder_layers.append(self.double_conv(prev_channels, feature))
            prev_channels = feature

        self.bottleneck = self.double_conv(features[-1], features[-1]*2)

        self.upconvs = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        rev_features = features[::-1]
        for feature in rev_features:
            self.upconvs.append(nn.ConvTranspose2d(features[-1]*2 if feature==rev_features[0] else feature*2, feature, kernel_size=2, stride=2))
            self.decoder_layers.append(self.double_conv(feature*2, feature))

        self.final_up = nn.ConvTranspose2d(features[0], features[0]//2, kernel_size=3, stride=3, padding=1, output_padding=2)
        self.final_conv = nn.Conv2d(features[0]//2, out_channels, kernel_size=1)

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
        encs = []
        for enc in self.encoder_layers:
            x = enc(x)
            encs.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip = encs[-(idx+1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder_layers[idx](x)
        x = self.final_up(x)
        x = self.final_conv(x)
        return x
    
    
class SRCNN_3x(nn.Module):
    """
    SRCNN adapted for direct 3x upsampling without pre-upsampling
    Takes LR input (e.g., 128x128) and outputs 3x larger (384x384)
    
    Architecture optimized for DEM super-resolution with 3x scaling
    """
    
    def __init__(self, num_channels=1):
        super(SRCNN_3x, self).__init__()
        
        # Feature extraction - deeper for better 3x upsampling
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Non-linear mapping - multiple layers for complex terrain features
        self.mapping = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling branch - generates 3x3 sub-pixels for each input pixel
        self.upsampling = nn.Sequential(
            nn.Conv2d(32, 32 * 9, kernel_size=3, padding=1),  # 9 = 3x3 upsampling
            nn.PixelShuffle(3),  # Rearranges to 3x upsampled output
        )
        
        # Reconstruction - refines the upsampled result
        self.reconstruction = nn.Conv2d(32, num_channels, kernel_size=3, padding=1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Input: e.g., (batch, 1, 128, 128)
        
        # Feature extraction
        features = self.feature_extraction(x)
        
        # Non-linear mapping
        mapped = self.mapping(features)
        
        # Upsampling using pixel shuffle
        upsampled = self.upsampling(mapped)  # Now (batch, 32, 384, 384)
        
        # Final reconstruction
        output = self.reconstruction(upsampled)  # (batch, 1, 384, 384)
        
        return output


class EnhancedSRCNN_3x(nn.Module):
    """
    Enhanced SRCNN with residual learning for direct 3x upsampling
    More comparable to your encoder-decoder architecture
    """
    
    def __init__(self, num_channels=1):
        super(EnhancedSRCNN_3x, self).__init__()
        
        # Initial feature extraction
        self.initial_conv = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        
        # Feature extraction blocks
        self.feature_blocks = nn.ModuleList([
            self._make_layer(32, 64, 3),
            self._make_layer(64, 128, 3), 
            self._make_layer(128, 256, 3),
        ])
        
        # Non-linear mapping with residual blocks
        self.mapping_blocks = nn.ModuleList([
            self._make_residual_block(256, 256),
            self._make_residual_block(256, 256),
            self._make_residual_block(256, 128),
            self._make_residual_block(128, 64),
        ])
        
        # Upsampling layers - progressive 3x upsampling
        self.upsample_conv = nn.Conv2d(64, 64 * 9, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(3)  # 3x upsampling
        
        # Reconstruction layers
        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_channels, kernel_size=3, padding=1)
        )
        
        # Upsampling for skip connection
        self.skip_upsample = nn.Upsample(scale_factor=3, mode='bicubic', align_corners=False)
        
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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Store input for skip connection
        input_skip = self.skip_upsample(x)
        
        # Initial feature extraction
        x = F.relu(self.initial_conv(x))
        
        # Progressive feature extraction
        for block in self.feature_blocks:
            x = block(x)
        
        # Non-linear mapping with residuals
        for block in self.mapping_blocks:
            identity = x
            out = block(x)
            # Handle channel dimension changes
            if out.shape[1] != identity.shape[1]:
                identity = F.adaptive_avg_pool2d(identity, 1)
                identity = F.interpolate(identity, size=out.shape[2:])
                identity = torch.cat([identity] * (out.shape[1] // identity.shape[1]), dim=1)
            x = F.relu(out + identity)
        
        # Upsampling
        x = self.upsample_conv(x)
        x = self.pixel_shuffle(x)  # Now 3x upsampled
        
        # Reconstruction
        x = self.reconstruction(x)
        
        # Add skip connection (residual from upsampled input)
        return x + input_skip
    

def edge_preserving_loss(pred, target):
    """
    Compute edge-preserving loss using Sobel operators.
    
    Args:
        pred: model predictions
        target: ground truth
        
    Returns:
        edge_loss: edge-preserving loss
    """
    # Sobel operators for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    
    pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
    pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
    target_edge_x = F.conv2d(target, sobel_x, padding=1)
    target_edge_y = F.conv2d(target, sobel_y, padding=1)
    
    edge_loss = F.mse_loss(pred_edge_x, target_edge_x) + F.mse_loss(pred_edge_y, target_edge_y)
    return edge_loss

def gradient_loss(pred, target):
    # Compute gradients in x and y directions
    pred_grad_x = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_grad_y = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_grad_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_grad_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    
    grad_loss_x = F.mse_loss(pred_grad_x, target_grad_x)
    grad_loss_y = F.mse_loss(pred_grad_y, target_grad_y)
    return grad_loss_x + grad_loss_y