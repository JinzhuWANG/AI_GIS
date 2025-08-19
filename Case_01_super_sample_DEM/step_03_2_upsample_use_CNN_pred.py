import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from affine import Affine
import warnings
warnings.filterwarnings('ignore')

class ChunkDataset(Dataset):
    """Dataset for processing chunks of 90m DEM data"""
    def __init__(self, dem_90m, chunk_size=128):
        self.chunk_size = chunk_size
        self.chunks = []
        self.chunk_positions = []
        
        # Get data dimensions
        height, width = dem_90m.shape
        print(f"Processing {height}x{width} data into {chunk_size}x{chunk_size} chunks")
        
        # Create chunks by sliding window
        for i in range(0, height, chunk_size):
            for j in range(0, width, chunk_size):
                end_i = min(i + chunk_size, height)
                end_j = min(j + chunk_size, width)
                
                # Extract chunk using isel
                chunk = dem_90m.isel(y=slice(i, end_i), x=slice(j, end_j))
                
                if chunk.size > 0:  # Skip empty chunks
                    # Get the actual data array
                    data = chunk.values.astype(np.float32)
                    h, w = data.shape
                    
                    # If chunk is smaller than target size, pad it
                    if h < chunk_size or w < chunk_size:
                        padded_data = np.zeros((chunk_size, chunk_size), dtype=np.float32)
                        padded_data[:h, :w] = data
                        data = padded_data
                    elif h > chunk_size or w > chunk_size:
                        # Crop to target size
                        data = data[:chunk_size, :chunk_size]
                    
                    # Normalize the chunk (min-max normalization like in training)
                    data_min, data_max = data.min(), data.max()
                    if data_max > data_min:
                        normalized_data = (data - data_min) / (data_max - data_min)
                    else:
                        normalized_data = data.copy()
                    
                    self.chunks.append(normalized_data)
                    self.chunk_positions.append({
                        'y_start': i,
                        'x_start': j,
                        'y_end': end_i,
                        'x_end': end_j,
                        'original_shape': (h, w),
                        'min_val': float(data_min),
                        'max_val': float(data_max)
                    })
        
        print(f"Created dataset with {len(self.chunks)} chunks of size {chunk_size}x{chunk_size}")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        # Return tensor with channel dimension and position info
        chunk_tensor = torch.FloatTensor(self.chunks[idx]).unsqueeze(0)  # Add channel dim
        return chunk_tensor, self.chunk_positions[idx]

def chunk_based_inference(model, dem_90m, batch_size=8, chunk_size=128):
    """
    Perform inference using chunked data approach following step 3-1 model requirements
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    print(f"Original 90m DEM shape: {dem_90m.shape}")
    
    # Create the output 30m structure (3x larger)
    output_height = dem_90m.shape[0] * 3
    output_width = dem_90m.shape[1] * 3
    
    print(f"Creating empty 30m DEM structure: {output_height} x {output_width}")
    
    # Create coordinates for 30m data
    x_90m = dem_90m.x.values
    y_90m = dem_90m.y.values
    x_min, x_max = float(x_90m[0]), float(x_90m[-1])
    y_min, y_max = float(y_90m[0]), float(y_90m[-1])
    
    # Create 30m coordinate arrays
    x_30m = np.linspace(x_min, x_max, output_width)
    y_30m = np.linspace(y_min, y_max, output_height)
    
    # Initialize empty 30m data array
    dem_30m_data = np.zeros((output_height, output_width), dtype=np.float32)
    
    # Create dataset (this handles chunking internally)
    dataset = ChunkDataset(dem_90m, chunk_size)
    
    # Simple dataloader that handles tensors properly
    def simple_collate_fn(batch):
        chunks = torch.stack([item[0] for item in batch])
        positions = [item[1] for item in batch]
        return chunks, positions
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, collate_fn=simple_collate_fn)
    
    print(f"Processing {len(dataset)} chunks in batches of {batch_size}")
    
    processed_chunks = 0
    total_batches = len(dataloader)
    
    with torch.no_grad():
        for batch_idx, (chunk_batch, position_batch) in enumerate(dataloader):
            if batch_idx % max(1, total_batches // 10) == 0:
                print(f"Processing batch {batch_idx+1}/{total_batches} ({(batch_idx+1)/total_batches*100:.1f}%)")
            
            # Move to device - chunks already have channel dimension
            chunk_batch = chunk_batch.to(device)  # [batch, 1, 128, 128]
            
            # Perform batch inference (128x128 -> 416x416)
            predictions = model(chunk_batch)  # [batch, 1, 416, 416]
            
            # Clip the 32-pixel buffer to get the original 384x384 output
            buffer_size = 16  # 32 / 2 = 16 pixels on each side
            predictions = predictions[:, :, buffer_size:-buffer_size, buffer_size:-buffer_size]  # [batch, 1, 384, 384]
            predictions = predictions.cpu().numpy().squeeze()  # Remove channel dimension
            
            # Handle single item batch
            if predictions.ndim == 2:
                predictions = predictions[np.newaxis, :]
            
            # Process each prediction and map to 30m grid
            current_batch_size = len(chunk_batch)
            for i in range(current_batch_size):
                prediction = predictions[i]  # 384x384
                pos_info = position_batch[i]  # Dictionary with position info
                
                # Denormalize prediction
                min_val = pos_info['min_val']
                max_val = pos_info['max_val']
                if max_val > min_val:
                    prediction = prediction * (max_val - min_val) + min_val
                
                # Calculate where this prediction should go in the 30m grid
                y_start_90m = pos_info['y_start']
                x_start_90m = pos_info['x_start']
                orig_h, orig_w = pos_info['original_shape']
                
                # Map to 30m coordinates (3x scaling)
                start_y_30m = y_start_90m * 3
                start_x_30m = x_start_90m * 3
                
                # Calculate the actual size of the prediction to use
                pred_h_to_use = min(384, orig_h * 3)
                pred_w_to_use = min(384, orig_w * 3)
                
                # Make sure we don't exceed the output boundaries
                end_y_30m = min(start_y_30m + pred_h_to_use, output_height)
                end_x_30m = min(start_x_30m + pred_w_to_use, output_width)
                
                # Calculate actual dimensions to copy
                actual_h = end_y_30m - start_y_30m
                actual_w = end_x_30m - start_x_30m
                
                # Place prediction in 30m grid
                if actual_h > 0 and actual_w > 0:
                    dem_30m_data[start_y_30m:end_y_30m, start_x_30m:end_x_30m] = \
                        prediction[:actual_h, :actual_w]
            
            processed_chunks += current_batch_size
    
    print(f"Processed {processed_chunks} chunks total")
    
    # Create the final 30m xarray DataArray
    dem_30m = xr.DataArray(
        dem_30m_data,
        coords={
            'y': y_30m,
            'x': x_30m
        },
        dims=['y', 'x'],
        name='data',
        attrs=dem_90m.attrs.copy()
    )
    
    return dem_30m

if __name__ == "__main__":
    print("Loading data and model...")
    
    # Load only the 90m DEM data with chunking
    dem_90m = xr.open_dataarray('data/DEM_90m.nc', chunks=256)
    
    print(f"90m DEM shape: {dem_90m.shape}")
    
    # Load the trained model using TorchScript
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load the traced model
    model = torch.jit.load('data/CNN_super_resolution_traced.pt', map_location=device)
    print("TorchScript traced model loaded successfully!")
    
    model.eval()
    print(f"Model moved to device: {device}")
    
    # Perform chunk-based super-resolution inference
    print("\nStarting CNN super-resolution inference with chunk-based processing...")
    
    # Determine optimal batch size based on available memory
    if device.type == "cuda":
        batch_size = 16
    else:
        batch_size = 4
    
    print(f"Using batch size: {batch_size}")
    
    # Use the chunk-based inference
    dem_30m_cnn = chunk_based_inference(
        model, 
        dem_90m, 
        batch_size=batch_size,
        chunk_size=128
    )
    
    print(f"Super-resolution prediction shape: {dem_30m_cnn.shape}")
    print(f"Original 90m shape: {dem_90m.shape}")
    print(f"Resolution increase: {dem_30m_cnn.shape[0]/dem_90m.shape[0]:.1f}x")
    
    # Copy CRS information from 90m DEM if available
    if hasattr(dem_90m, 'rio'):
        dem_30m_cnn.rio.write_crs(dem_90m.rio.crs, inplace=True)
        
        # Calculate new transform for the upsampled data (3x finer resolution)
        old_transform = dem_90m.rio.transform()
        new_transform = Affine(
            old_transform.a / 3, old_transform.b, old_transform.c,
            old_transform.d, old_transform.e / 3, old_transform.f
        )
        dem_30m_cnn.rio.write_transform(new_transform, inplace=True)
    
    # Save the result
    print("\nSaving CNN prediction result...")
    dem_30m_cnn.astype(np.uint16).to_netcdf(
        'data/DEM_30m_CNN_prediction.nc',
        engine='netcdf4',
        encoding={
            'data': {
                'dtype': 'uint16',
                'zlib': True,
                'complevel': 6,
                'chunksizes': (256, 256)
            }
        }
    )
    
    print("CNN prediction saved as 'data/DEM_30m_CNN_prediction.nc'")
    
    # Calculate and display statistics
    print(f"\nStatistics comparison:")
    print(f"90m DEM - Min: {dem_90m.min().values:.2f}, Max: {dem_90m.max().values:.2f}, Mean: {dem_90m.mean().values:.2f}")
    print(f"CNN Prediction - Min: {dem_30m_cnn.min().values:.2f}, Max: {dem_30m_cnn.max().values:.2f}, Mean: {dem_30m_cnn.mean().values:.2f}")
    print(f"Upsampling ratio: {dem_30m_cnn.shape[0] / dem_90m.shape[0]:.1f}x in each dimension")
    
    print("\nInference completed successfully!")