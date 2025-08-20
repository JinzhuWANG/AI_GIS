import csv
import os
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

from helper import EnhancedSRCNN_4x, edge_preserving_loss, gradient_loss

# Initialize the model (now adapted for RGB images with 64-32-64-128-256 architecture)
model = EnhancedSRCNN_4x()
NUM_EPOCH = 1000

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
    print(f"Preparing RGB data...")
    print(f"High-res shape: {high_res_data.shape}")
    print(f"Low-res shape: {low_res_data.shape}")
    
    # Convert from (N, H, W, C) to (N, C, H, W) format for PyTorch
    high_res_torch = np.transpose(high_res_data, (0, 3, 1, 2))
    low_res_torch = np.transpose(low_res_data, (0, 3, 1, 2))
    
    # Normalize to [0, 1] range (assuming input is uint8 [0, 255])
    high_res_torch = high_res_torch.astype(np.float32) / 255.0
    low_res_torch = low_res_torch.astype(np.float32) / 255.0
    
    print(f"Normalized high-res range: [{high_res_torch.min():.3f}, {high_res_torch.max():.3f}]")
    print(f"Normalized low-res range: [{low_res_torch.min():.3f}, {low_res_torch.max():.3f}]")
    
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    train_losses = []
    val_losses = []
    
    # Write header if file does not exist
    metrics_csv_path = 'data/performance_metrics.csv'
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate'])
    
    # Check model output dimensions
    with torch.no_grad():
        sample_batch = next(iter(train_loader))
        sample_low_res = sample_batch[0][:1].to(device)
        sample_high_res = sample_batch[1][:1].to(device)
        sample_output = model(sample_low_res)
        print(f"Input shape: {sample_low_res.shape}")
        print(f"Target shape: {sample_high_res.shape}")
        print(f"Model output shape: {sample_output.shape}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 50
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (low_res, high_res) in enumerate(train_loader):
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            optimizer.zero_grad()
            outputs = model(low_res)
            
            # Combined loss
            mse_loss = criterion(outputs, high_res)
            edge_loss = edge_preserving_loss(outputs, high_res)
            grad_loss = gradient_loss(outputs, high_res)
            
            total_loss = mse_loss + 0.1 * edge_loss + 0.1 * grad_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                outputs = model(low_res)
                val_loss += criterion(outputs, high_res).item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Write metrics to CSV for each epoch
        with open(metrics_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, current_lr])
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'data/best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.8f}')
            
            # Print GPU memory usage if using CUDA
            if device.type == "cuda":
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f'GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB')

            # Save TorchScript traced model periodically
            model.eval()
            sample_input = torch.randn(1, 3, 64, 64).to(device)  # RGB input
            traced_model = torch.jit.trace(model, sample_input)
            traced_model_path = f'data/RGB_CNN_super_resolution_traced_epoch_{epoch+1:03d}.pt'
            traced_model.save(traced_model_path)
            model.train()
            print(f'Traced model saved as {traced_model_path}')
    
    return train_losses, val_losses


if __name__ == "__main__":
    print("Loading RGB image data...")
    
    # Load the RGB image datasets created in step 1
    high_res_data = xr.open_dataarray('data/train.nc')  # 256x256 images
    low_res_data = xr.open_dataarray('data/train_coarse2.nc')  # 64x64 images
    
    print(f"High-resolution data shape: {high_res_data.shape}")
    print(f"Low-resolution data shape: {low_res_data.shape}")
    
    # Convert xarray to numpy
    high_res_np = high_res_data.values  # (N_samples, height, width, channels)
    low_res_np = low_res_data.values
    
    # Prepare data for training
    high_res_torch, low_res_torch = prepare_rgb_data(high_res_np, low_res_np)
    
    print(f"Prepared high-res shape: {high_res_torch.shape}")
    print(f"Prepared low-res shape: {low_res_torch.shape}")
    
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create and train the model
    print("\nCreating RGB CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    print("\nStarting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCH)
    
    # Save the final TorchScript traced model
    model.eval()
    sample_input = torch.randn(1, 3, 64, 64).to(device)
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.save('data/RGB_CNN_super_resolution_traced_final.pt')
    print("\nFinal TorchScript traced model saved as 'data/RGB_CNN_super_resolution_traced_final.pt'")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.yscale('log')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-100:], label='Training Loss (Last 100)')
    plt.plot(val_losses[-100:], label='Validation Loss (Last 100)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Progress (Last 100 Epochs)')
    
    plt.tight_layout()
    plt.savefig('data/RGB_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nFinal Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"Best Validation Loss: {min(val_losses):.6f}")
    print("Training completed!")