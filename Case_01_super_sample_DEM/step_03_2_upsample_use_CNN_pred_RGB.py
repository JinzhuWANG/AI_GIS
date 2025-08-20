import xarray as xr
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class RGBPredictionDataset(Dataset):
    """Dataset for RGB image prediction"""
    def __init__(self, low_res_images):
        # Convert from (N, H, W, C) to (N, C, H, W) format for PyTorch
        self.low_res_images = np.transpose(low_res_images, (0, 3, 1, 2))
        # Normalize to [0, 1] range
        self.low_res_images = self.low_res_images.astype(np.float32) / 255.0
        
    def __len__(self):
        return len(self.low_res_images)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.low_res_images[idx])

def predict_rgb_super_resolution(low_res_data, model_path, batch_size=8, save_path=None):
    """
    Use trained CNN model to predict high-resolution RGB images
    
    Args:
        low_res_data: numpy array of shape (N, H, W, C) with low-resolution images
        model_path: path to the trained CNN model (.pt file)
        batch_size: batch size for prediction
        save_path: optional path to save the predicted images
    
    Returns:
        numpy array of predicted high-resolution images (N, H_high, W_high, C)
    """
    print(f"Loading CNN model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the traced model
    try:
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Create dataset and dataloader
    dataset = RGBPredictionDataset(low_res_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Predicting on {len(dataset)} images with batch size {batch_size}")
    
    predictions = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Model prediction
            pred_batch = model(batch)
            pred_batch = pred_batch.cpu().numpy()
            
            predictions.append(pred_batch)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Convert back to (N, H, W, C) format and scale to [0, 255]
    predictions = np.transpose(predictions, (0, 2, 3, 1))
    predictions = np.clip(predictions * 255, 0, 255).astype(np.uint8)
    
    print(f"Prediction completed. Output shape: {predictions.shape}")
    
    # Save predictions if path provided
    if save_path:
        # Create xarray DataArray with proper coordinates
        n_samples, height, width, channels = predictions.shape
        
        pred_xr = xr.DataArray(
            predictions,
            dims=['sample', 'height', 'width', 'channels'],
            coords={
                'sample': range(n_samples),
                'height': range(height),
                'width': range(width),
                'channels': ['r', 'g', 'b']
            }
        )
        
        # Save with compression
        pred_xr.name = 'data'
        encoding = {
            'data': {
                'dtype': 'uint8',
                'zlib': True,
                'complevel': 6,
            }
        }
        
        pred_xr.to_netcdf(save_path, encoding=encoding)
        print(f"Predictions saved to {save_path}")
    
    return predictions

def visualize_predictions(original_images, low_res_images, predicted_images, 
                         n_samples=3, save_path=None):
    """
    Create a visualization comparing original, low-res, and predicted images
    """
    np.random.seed(42)
    
    # Select random samples for visualization
    n_total = original_images.shape[0]
    sample_indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    titles = ['Low Resolution (64×64)', 'CNN Prediction (256×256)', 'Original (256×256)']
    
    for i, idx in enumerate(sample_indices):
        # Low resolution image (upscaled for display)
        low_res = low_res_images[idx]
        # Use nearest neighbor upscaling for comparison
        low_res_display = np.repeat(np.repeat(low_res, 4, axis=0), 4, axis=1)
        
        # Predicted high resolution
        pred_high_res = predicted_images[idx]
        
        # Original high resolution
        orig_high_res = original_images[idx]
        
        images = [low_res_display, pred_high_res, orig_high_res]
        
        for j, (img, title) in enumerate(zip(images, titles)):
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{title}\\nSample {idx}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()

def calculate_prediction_metrics(original_images, predicted_images):
    """Calculate simple metrics for the predictions"""
    # Flatten arrays
    orig_flat = original_images.flatten().astype(np.float32)
    pred_flat = predicted_images.flatten().astype(np.float32)
    
    # MSE and RMSE
    mse = np.mean((orig_flat - pred_flat) ** 2)
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(orig_flat - pred_flat))
    
    # PSNR
    psnr = 20 * np.log10(255) - 10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"Prediction Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'PSNR': psnr}

if __name__ == "__main__":
    print("Step 3-2: RGB Image Super-Resolution Prediction...")
    
    # Load the test data
    print("\n1. Loading test data...")
    
    # Load low-resolution test images (64x64)
    low_res_test = xr.open_dataarray('data/test_coarse2.nc')
    print(f"Low-resolution test images shape: {low_res_test.shape}")
    
    # Load high-resolution test images for comparison (256x256)
    high_res_test = xr.open_dataarray('data/test.nc')
    print(f"High-resolution test images shape: {high_res_test.shape}")
    
    # Convert to numpy arrays
    low_res_np = low_res_test.values
    high_res_np = high_res_test.values
    
    # Check if we have test data
    if low_res_np.shape[0] == 0:
        print("No test data available. Using training data for demonstration...")
        # Load training data instead
        low_res_train = xr.open_dataarray('data/train_coarse2.nc')
        high_res_train = xr.open_dataarray('data/train.nc')
        
        # Use a subset for testing
        n_test = min(10, low_res_train.shape[0])
        low_res_np = low_res_train.values[:n_test]
        high_res_np = high_res_train.values[:n_test]
        print(f"Using {n_test} training samples for demonstration")
    
    print(f"Using {low_res_np.shape[0]} images for prediction")
    
    # Predict using the trained CNN model
    print("\n2. Making CNN predictions...")
    
    model_path = 'data/RGB_CNN_super_resolution_traced_final.pt'
    
    predicted_images = predict_rgb_super_resolution(
        low_res_np, 
        model_path, 
        batch_size=4,
        save_path='data/RGB_CNN_predictions.nc'
    )
    
    if predicted_images is not None:
        print(f"Prediction successful! Shape: {predicted_images.shape}")
        
        # Calculate metrics
        print("\n3. Calculating prediction metrics...")
        metrics = calculate_prediction_metrics(high_res_np, predicted_images)
        
        # Create visualizations
        print("\n4. Creating visualizations...")
        
        # Sample visualization
        visualize_predictions(
            high_res_np, 
            low_res_np, 
            predicted_images,
            n_samples=3,
            save_path='data/RGB_prediction_comparison.png'
        )
        
        # Create difference maps
        print("\n5. Creating difference analysis...")
        
        # Calculate pixel-wise differences
        diff_images = np.abs(high_res_np.astype(np.float32) - predicted_images.astype(np.float32))
        
        # Visualize difference for a sample
        sample_idx = 0
        if high_res_np.shape[0] > sample_idx:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Original
            axes[0].imshow(high_res_np[sample_idx])
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Prediction
            axes[1].imshow(predicted_images[sample_idx])
            axes[1].set_title('CNN Prediction')
            axes[1].axis('off')
            
            # Difference (grayscale)
            diff_gray = np.mean(diff_images[sample_idx], axis=2)
            im = axes[2].imshow(diff_gray, cmap='hot', vmin=0, vmax=50)
            axes[2].set_title('Absolute Difference')
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2])
            
            # Histogram of differences
            axes[3].hist(diff_gray.flatten(), bins=50, alpha=0.7, color='red')
            axes[3].set_title('Difference Distribution')
            axes[3].set_xlabel('Absolute Pixel Difference')
            axes[3].set_ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('data/RGB_difference_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        print(f"\n6. Summary:")
        print("="*50)
        print(f"Successfully predicted {predicted_images.shape[0]} high-resolution RGB images")
        print(f"Input resolution: {low_res_np.shape[1]}×{low_res_np.shape[2]}")
        print(f"Output resolution: {predicted_images.shape[1]}×{predicted_images.shape[2]}")
        print(f"Upscaling factor: {predicted_images.shape[1] // low_res_np.shape[1]}×")
        print(f"PSNR: {metrics['PSNR']:.2f} dB")
        
        print("\nGenerated files:")
        print("- data/RGB_CNN_predictions.nc")
        print("- data/RGB_prediction_comparison.png")
        print("- data/RGB_difference_analysis.png")
        
    else:
        print("Prediction failed! Please check that the model file exists and is valid.")
        print(f"Expected model path: {model_path}")
