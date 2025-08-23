import os
import xarray as xr
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the traced model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    # Create dataset and dataloader
    dataset = RGBPredictionDataset(low_res_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    
    predictions = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Predicting', unit='batch')):
            batch = batch.to(device)
            
            # Model prediction
            pred_batch = model(batch)
            pred_batch = pred_batch.cpu().numpy()
            
            predictions.append(pred_batch)
    
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



if __name__ == "__main__":
    
    os.makedirs('data/images/CNN_upsample/test', exist_ok=True)
    
    # Save to nc
    low_res_test = xr.open_dataarray('data/images/test/coarse/test_coarse2.nc')
    model_path = 'data/images/RGB_CNN_super_resolution_traced_final.pt'
    predicted_images = predict_rgb_super_resolution(
        low_res_test.values, 
        model_path, 
        batch_size=4,
        save_path='data/images/CNN_upsample/test/RGB_CNN_predictions.nc'
    )
    
    # Save to png
    for i, img in enumerate(predicted_images):
        img = Image.fromarray(img)
        img.save(f"data/images/CNN_upsample/test/RGB_CNN_prediction_{i:03d}.png")
