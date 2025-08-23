import csv
import os
import random
import xarray as xr
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from helper import( 
    SuperResolutionCNN_shallow,
    SuperResolutionCNN_deep,
    edge_preserving_loss
)

# Initialize the model (now adapted for RGB images with 64-32-64-128-256 architecture)
model = SuperResolutionCNN_shallow()
MAX_EPOCH = 100        # Number of training epochs
NUM_WORKERS = 0        # Set to 0 to avoid multiprocessing issues in some environments

# Define the root directory for saving metrics and models
SAVE_PATH = 'data/images/CNN_shallow'
os.makedirs(SAVE_PATH, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class RGBImageDataset(Dataset):
    """Dataset for RGB image super-resolution training"""
    def __init__(self, low_res_images, high_res_images):
        self.low_res_images = torch.FloatTensor(low_res_images)
        self.high_res_images = torch.FloatTensor(high_res_images)
    
    def __len__(self):
        return len(self.low_res_images)
    
    def __getitem__(self, idx):
        return self.low_res_images[idx], self.high_res_images[idx]


def prepare_rgb_data(high_res_data, low_res_data):
    """
    Prepare RGB image data for training
    Input: high_res_data (N, H, W, C), low_res_data (N, H_low, W_low, C)
    Output: normalized tensors in (N, C, H, W) format
    """

    # Convert from (N, H, W, C) to (N, C, H, W) format for PyTorch
    high_res_torch = np.transpose(high_res_data, (0, 3, 1, 2))
    low_res_torch = np.transpose(low_res_data, (0, 3, 1, 2))
    # Normalize to [0, 1] range (assuming input is uint8 [0, 255])
    high_res_torch = high_res_torch.astype(np.float32) / 255.0
    low_res_torch = low_res_torch.astype(np.float32) / 255.0

    return high_res_torch, low_res_torch


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the CNN model for RGB image super-resolution
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Track the best validation loss
    
    # Write header if file does not exist
    metrics_csv_path = os.path.join(SAVE_PATH, 'performance_metrics.csv')
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss'])
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Training phase with progress bar for batches
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc='Training', unit='batch', leave=False)
        
        for batch_idx, (low_res, high_res) in enumerate(train_pbar):
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            optimizer.zero_grad()
            outputs = model(low_res)
            
            # Combined loss
            mse_loss = criterion(outputs, high_res) #+ 0.5 * edge_preserving_loss(outputs, high_res)
            mse_loss.backward()
            optimizer.step()
            
            train_loss += mse_loss.item()
            
            # Update training progress bar
            current_train_loss = train_loss / (batch_idx + 1)
            train_pbar.set_postfix({
                'Loss': f'{mse_loss.item():.6f}',
                'Avg Loss': f'{current_train_loss:.6f}'
            })
        
        train_pbar.close()
        
        # Validation phase with progress bar
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(val_loader, desc='Validation', unit='batch', leave=False)
        
        with torch.no_grad():
            for batch_idx, (low_res, high_res) in enumerate(val_pbar):
                low_res, high_res = low_res.to(device), high_res.to(device)
                outputs = model(low_res)
                batch_val_loss = criterion(outputs, high_res).item() #+ 0.5 * edge_preserving_loss(outputs, high_res).item()
                val_loss += batch_val_loss
                
                # Update validation progress bar
                current_val_loss = val_loss / (batch_idx + 1)
                val_pbar.set_postfix({
                    'Loss': f'{batch_val_loss:.6f}',
                    'Avg Loss': f'{current_val_loss:.6f}'
                })
        
        val_pbar.close()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        
        # Write metrics to CSV for each epoch
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss])
        
        # Save best model only if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.eval()
            sample_input = torch.randn(1, 3, 64, 64).to(device)  # RGB input
            traced_model = torch.jit.trace(model, sample_input)
            traced_model.save(os.path.join(SAVE_PATH, 'best_model_traced.pt'))
            model.train()  # Switch back to training mode
            
            print(f"✓ New best model saved! Val Loss: {val_loss:.6f}")
        else:
            print(f"Val Loss: {val_loss:.6f} (Best: {best_val_loss:.6f})")
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Save TorchScript traced model periodically (every 50 epochs)
        if (epoch + 1) % 50 == 0:
            model.eval()
            sample_input = torch.randn(1, 3, 64, 64).to(device)  # RGB input
            traced_model = torch.jit.trace(model, sample_input)
            traced_model_path = os.path.join(SAVE_PATH, f'RGB_CNN_super_resolution_traced_epoch_{epoch+1:03d}.pt')
            traced_model.save(traced_model_path)
            model.train()
            print(f'Traced model saved as {traced_model_path}')
            
            # Print GPU memory usage if using CUDA
            if device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB')
    
    return train_losses, val_losses


if __name__ == "__main__":
    print("Loading RGB image data...")
    
    # Load the RGB image datasets created in step 1
    high_res_data = xr.open_dataarray('data/images/train/original/train.nc')  # 256x256 images
    low_res_data = xr.open_dataarray('data/images/train/coarse/train_coarse.nc')  # 64x64 images

    # Convert xarray to numpy
    high_res_np = high_res_data.values  # (N_samples, height, width, channels)
    low_res_np = low_res_data.values
    high_res_torch, low_res_torch = prepare_rgb_data(high_res_np, low_res_np)
    
    # Split into training and validation sets (80/20 split)
    train_ratio = 0.8
    n_samples = high_res_torch.shape[0]
    n_train = int(n_samples * train_ratio)
    
    # Random shuffle and split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_high = high_res_torch[train_indices]
    train_low = low_res_torch[train_indices]
    val_high = high_res_torch[val_indices]
    val_low = low_res_torch[val_indices]
    
    print(f"\nTraining set: {train_high.shape[0]} samples")
    print(f"Validation set: {val_high.shape[0]} samples")
    
    # Create datasets and data loaders
    train_dataset = RGBImageDataset(train_low, train_high)
    val_dataset = RGBImageDataset(val_low, val_high)
    
    batch_size = 8  # Adjust based on GPU memory
    # Set num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    # Create and train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\nStarting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=MAX_EPOCH)
    
    # Save the final TorchScript traced model
    model.eval()
    sample_input = torch.randn(1, 3, 64, 64).to(device)
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.save(os.path.join(SAVE_PATH, 'RGB_CNN_super_resolution_traced_final.pt'))
