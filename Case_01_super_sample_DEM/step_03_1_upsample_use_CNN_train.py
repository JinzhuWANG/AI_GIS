import xarray as xr
import numpy as np
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class DEMDataset(Dataset):
    def __init__(self, low_res_samples, high_res_samples):
        self.low_res_samples = torch.FloatTensor(low_res_samples)
        self.high_res_samples = torch.FloatTensor(high_res_samples)
    
    def __len__(self):
        return len(self.low_res_samples)
    
    def __getitem__(self, idx):
        return self.low_res_samples[idx], self.high_res_samples[idx]

class SuperResolutionCNN(nn.Module):
    def __init__(self):
        super(SuperResolutionCNN, self).__init__()
        
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
        
        # Super-resolution: 128x128 -> (384+32)x(384+32) = 416x416
        # Added 32-pixel buffer to reduce edge effects
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=3, padding=1+11, output_padding=2)
        
    def forward(self, x):
        # Encoder: 128 -> 64 -> 32
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Decoder: 32 -> 64 -> 128 -> 384
        x = F.relu(self.bn3(self.deconv1(x)))
        x = F.relu(self.bn4(self.deconv2(x)))
        x = self.deconv3(x)
        
        return x

def create_samples(dem_30m, dem_90m, num_samples=500, low_res_size=128, high_res_size=416):
    """
    Create training samples from DEM data
    low_res_size: size of input samples (128x128)
    high_res_size: size of target samples (384x384)
    """
    print(f"Creating {num_samples} samples of size {low_res_size}x{low_res_size} -> {high_res_size}x{high_res_size}")
    
    # Get data dimensions
    height_30m, width_30m = dem_30m.shape
    height_90m, width_90m = dem_90m.shape
    
    # Calculate the ratio between 30m and 90m data (should be 3x)
    ratio_y = height_30m // height_90m
    ratio_x = width_30m // width_90m
    
    print(f"30m DEM shape: {height_30m} x {width_30m}")
    print(f"90m DEM shape: {height_90m} x {width_90m}")
    print(f"Ratio: {ratio_y}x{ratio_x}")
    
    low_res_samples = []
    high_res_samples = []
    
    # Calculate safe sampling range - be more conservative
    # For 30m data, we need 384x384 patches, so we need at least 192 pixels from edge
    safe_margin_30m = high_res_size // 2
    # For 90m data, we need enough space for the low-res patch
    safe_margin_90m = (low_res_size // 2) // ratio_y
    
    # Ensure we have enough data to sample from
    if height_30m < high_res_size or width_30m < high_res_size:
        raise ValueError(f"30m DEM too small ({height_30m}x{width_30m}) for {high_res_size}x{high_res_size} patches")
    
    if height_90m < (low_res_size // ratio_y) or width_90m < (low_res_size // ratio_x):
        raise ValueError(f"90m DEM too small ({height_90m}x{width_90m}) for required patches")
    
    max_y_30m = height_30m - safe_margin_30m
    max_x_30m = width_30m - safe_margin_30m
    max_y_90m = height_90m - safe_margin_90m
    max_x_90m = width_90m - safe_margin_90m
    
    print(f"Safe sampling range - 30m: ({safe_margin_30m}, {safe_margin_30m}) to ({max_y_30m}, {max_x_30m})")
    print(f"Safe sampling range - 90m: ({safe_margin_90m}, {safe_margin_90m}) to ({max_y_90m}, {max_x_90m})")
    
    successful_samples = 0
    for i in range(num_samples * 2):  # Try more iterations to get enough valid samples
        if successful_samples >= num_samples:
            break
            
        # Randomly sample center coordinates for 90m data
        center_y_90m = random.randint(safe_margin_90m, max_y_90m)
        center_x_90m = random.randint(safe_margin_90m, max_x_90m)
        
        # Calculate corresponding center for 30m data
        center_y_30m = center_y_90m * ratio_y
        center_x_30m = center_x_90m * ratio_x
        
        # Extract samples
        half_low_size = low_res_size // 2
        half_high_size = high_res_size // 2
        half_size_90m = half_low_size // ratio_y
        
        # Bounds checking for 30m data
        if (center_y_30m - half_high_size < 0 or center_y_30m + half_high_size > height_30m or
            center_x_30m - half_high_size < 0 or center_x_30m + half_high_size > width_30m):
            continue
            
        # Bounds checking for 90m data  
        if (center_y_90m - half_size_90m < 0 or center_y_90m + half_size_90m > height_90m or
            center_x_90m - half_size_90m < 0 or center_x_90m + half_size_90m > width_90m):
            continue
        
        try:
            # Low resolution sample (90m data upsampled to 128x128)
            low_res_patch = dem_90m[
                center_y_90m - half_size_90m:center_y_90m + half_size_90m,
                center_x_90m - half_size_90m:center_x_90m + half_size_90m
            ]
            
            # Check if patch is valid
            if low_res_patch.size == 0:
                continue
            
            # Upsample low resolution patch to 128x128 using bilinear interpolation
            low_res_patch_upsampled = low_res_patch.interp(
                y=np.linspace(float(low_res_patch.y[0]), float(low_res_patch.y[-1]), low_res_size),
                x=np.linspace(float(low_res_patch.x[0]), float(low_res_patch.x[-1]), low_res_size),
                method='linear'
            )
            
            # High resolution sample (30m data at 384x384)
            high_res_patch = dem_30m[
                center_y_30m - half_high_size:center_y_30m + half_high_size,
                center_x_30m - half_high_size:center_x_30m + half_high_size
            ]
            
            # Check if patch is valid
            if high_res_patch.size == 0:
                continue
            
            # Convert to numpy and check for valid data
            low_res_np = low_res_patch_upsampled.values.astype(np.float32)
            high_res_np = high_res_patch.values.astype(np.float32)
            
            # Skip if patches don't have the expected size
            if low_res_np.shape != (low_res_size, low_res_size) or high_res_np.shape != (high_res_size, high_res_size):
                continue
                
            # Skip if data contains NaN or invalid values
            if np.isnan(low_res_np).any() or np.isnan(high_res_np).any():
                continue
            
            # Normalize data (simple min-max normalization)
            low_res_min, low_res_max = low_res_np.min(), low_res_np.max()
            high_res_min, high_res_max = high_res_np.min(), high_res_np.max()
            
            if low_res_max > low_res_min:
                low_res_np = (low_res_np - low_res_min) / (low_res_max - low_res_min)
            if high_res_max > high_res_min:
                high_res_np = (high_res_np - high_res_min) / (high_res_max - high_res_min)
            
            # Add channel dimension (for CNN input)
            low_res_np = low_res_np[np.newaxis, :, :]
            high_res_np = high_res_np[np.newaxis, :, :]
            
            low_res_samples.append(low_res_np)
            high_res_samples.append(high_res_np)
            successful_samples += 1
            
            if (successful_samples) % 100 == 0:
                print(f"Created {successful_samples}/{num_samples} samples")
                
        except Exception as e:
            print(f"Error creating sample {i}: {e}")
            continue
    
    if successful_samples < num_samples:
        print(f"Warning: Only created {successful_samples} samples out of {num_samples} requested")
    
    return np.array(low_res_samples), np.array(high_res_samples)

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the CNN model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    buffer_size = 16  # 32/2 = 16 pixels on each side
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (low_res, high_res) in enumerate(train_loader):
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            optimizer.zero_grad()
            outputs = model(low_res)
            
            # Clip the 32-pixel buffer from outputs to match high_res size
            outputs_clipped = outputs[:, :, buffer_size:-buffer_size, buffer_size:-buffer_size]
            
            loss = criterion(outputs_clipped, high_res)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                outputs = model(low_res)
                
                # Clip the 32-pixel buffer for validation as well
                outputs_clipped = outputs[:, :, buffer_size:-buffer_size, buffer_size:-buffer_size]
                
                val_loss += criterion(outputs_clipped, high_res).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # Print GPU memory usage if using CUDA
            if device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB')
    
    return train_losses, val_losses



if __name__ == "__main__":
    print("Loading DEM data...")
    
    # Load the DEM data
    dem_30m_original = xr.open_dataarray('data/DEM_30m.nc', chunks=256)
    dem_90m = xr.open_dataarray('data/DEM_90m.nc', chunks=256)
    
    print(f"30m DEM shape: {dem_30m_original.shape}")
    print(f"90m DEM shape: {dem_90m.shape}")
    
    # Create 500 samples
    print("\nCreating samples...")
    low_res_samples, high_res_samples = create_samples(
        dem_30m_original, dem_90m, num_samples=500, low_res_size=128, high_res_size=416
    )
    
    print(f"Created samples - Low res shape: {low_res_samples.shape}")
    print(f"Created samples - High res shape: {high_res_samples.shape}")
    
    # Split into training (300) and validation (200) sets
    train_low, val_low, train_high, val_high = train_test_split(
        low_res_samples, high_res_samples, 
        train_size=300, test_size=200, random_state=42
    )
    
    print(f"\nTraining set: {train_low.shape[0]} samples")
    print(f"Validation set: {val_low.shape[0]} samples")
    
    # Create datasets and data loaders (smaller batch size due to larger images)
    train_dataset = DEMDataset(train_low, train_high)
    val_dataset = DEMDataset(val_low, val_high)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Create and train the model
    print("\nCreating CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = SuperResolutionCNN()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model moved to: {device}")
    
    print("\nStarting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=100)
    
    # Save the complete trained model (traditional way)
    torch.save(model, 'data/CNN_super_resolution.pth')
    print("\nComplete model saved as 'data/CNN_super_resolution.pth'")
    
    # Also save as TorchScript traced model (no class definition needed for loading)
    model.eval()
    sample_input = torch.randn(1, 1, 128, 128).to(device)
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.save('data/CNN_super_resolution_traced.pt')
    print("TorchScript traced model saved as 'data/CNN_super_resolution_traced.pt'")
    

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('data/training_curves.png')
    plt.show()
    
    print(f"\nFinal Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    