import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from affine import Affine
import warnings
warnings.filterwarnings('ignore')

class ChunkDataset(Dataset):
    """Dataset for processing chunks of 90m DEM data with buffer for edge effect reduction"""
    def __init__(self, dem_90m, chunk_size=128, buffer_size=8, stride=128):
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.chunks = []
        self.chunk_positions = []
        
        # Input size with buffer
        self.input_size = chunk_size + 2 * buffer_size  # 128 + 16 = 144
        
        # Get data dimensions
        height, width = dem_90m.shape
        print(f"Processing {height}x{width} data into {self.input_size}x{self.input_size} chunks (including {buffer_size}-pixel buffer)")
        
        # Create chunks by sliding window with stride
        for i in range(0, height, stride):
            for j in range(0, width, stride):
                # Calculate chunk region with buffer
                start_i = max(0, i - buffer_size)
                start_j = max(0, j - buffer_size)
                end_i = min(i + chunk_size + buffer_size, height)
                end_j = min(j + chunk_size + buffer_size, width)
                
                # Extract chunk using isel
                chunk = dem_90m.isel(y=slice(start_i, end_i), x=slice(start_j, end_j))
                
                if chunk.size > 0:  # Skip empty chunks
                    # Get the actual data array
                    data = chunk.values.astype(np.float32)
                    h, w = data.shape
                    
                    # Create a properly sized array with buffer
                    padded_data = np.zeros((self.input_size, self.input_size), dtype=np.float32)
                    
                    # Calculate padding needed (for edge chunks)
                    pad_top = buffer_size - (i - start_i)
                    pad_left = buffer_size - (j - start_j)
                    pad_bottom = (i + chunk_size + buffer_size) - end_i
                    pad_right = (j + chunk_size + buffer_size) - end_j
                    
                    # Place actual data in the padded array at the right position
                    y_offset = max(0, pad_top)
                    x_offset = max(0, pad_left)
                    padded_data[y_offset:y_offset+h, x_offset:x_offset+w] = data
                    
                    # Normalize the chunk (min-max normalization like in training)
                    data_min, data_max = padded_data.min(), padded_data.max()
                    if data_max > data_min:
                        normalized_data = (padded_data - data_min) / (data_max - data_min)
                    else:
                        normalized_data = padded_data.copy()
                    
                    self.chunks.append(normalized_data)
                    self.chunk_positions.append({
                        'y_start': i,  # Original grid position (without buffer)
                        'x_start': j,  # Original grid position (without buffer)
                        'y_end': min(i + chunk_size, height),  # End position (without buffer)
                        'x_end': min(j + chunk_size, width),   # End position (without buffer)
                        'has_buffer_top': pad_top <= 0,
                        'has_buffer_left': pad_left <= 0,
                        'has_buffer_bottom': pad_bottom <= 0,
                        'has_buffer_right': pad_right <= 0,
                        'buffer_size': buffer_size,
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

def chunk_based_inference(model, dem_90m, batch_size=8, chunk_size=128, buffer_size=8):
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
    
    # Create dataset with buffer (this handles chunking and buffering internally)
    dataset = ChunkDataset(dem_90m, chunk_size=chunk_size, buffer_size=buffer_size, stride=chunk_size)
    
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
            
            # Perform batch inference on buffered inputs (144x144 -> 432x432)
            predictions = model(chunk_batch)  # [batch, 1, 432, 432]
            predictions = predictions.cpu().numpy().squeeze()  # Remove channel dimension
            
            # Handle single item batch
            if predictions.ndim == 2:
                predictions = predictions[np.newaxis, :]
            
            # Process each prediction and map to 30m grid
            current_batch_size = len(chunk_batch)
            for i in range(current_batch_size):
                prediction = predictions[i]  # 432x432
                pos_info = position_batch[i]  # Dictionary with position info
                
                # Denormalize prediction
                min_val = pos_info['min_val']
                max_val = pos_info['max_val']
                if max_val > min_val:
                    prediction = prediction * (max_val - min_val) + min_val
                
                # Extract position info
                y_start_90m = pos_info['y_start']
                x_start_90m = pos_info['x_start']
                y_end_90m = pos_info['y_end']
                x_end_90m = pos_info['x_end']
                buffer_size = pos_info['buffer_size']
                
                # Buffer size in output space (3x scaling)
                buffer_size_30m = buffer_size * 3
                
                # Map to 30m coordinates (3x scaling)
                start_y_30m = y_start_90m * 3
                start_x_30m = x_start_90m * 3
                end_y_30m = y_end_90m * 3
                end_x_30m = x_end_90m * 3
                
                # Calculate the slice of the prediction to use (remove buffer)
                # Always use central part of the prediction
                pred_y_start = buffer_size_30m
                pred_x_start = buffer_size_30m
                pred_y_end = buffer_size_30m + (y_end_90m - y_start_90m) * 3
                pred_x_end = buffer_size_30m + (x_end_90m - x_start_90m) * 3
                
                # Make sure we don't exceed the output boundaries
                end_y_30m = min(end_y_30m, output_height)
                end_x_30m = min(end_x_30m, output_width)
                
                # Calculate actual dimensions to copy
                actual_h = end_y_30m - start_y_30m
                actual_w = end_x_30m - start_x_30m
                
                # Place prediction in 30m grid (central part only - without buffer)
                if actual_h > 0 and actual_w > 0:
                    # Take the slice without buffer from the prediction
                    prediction_slice = prediction[pred_y_start:pred_y_end, pred_x_start:pred_x_end]
                    # Handle cases where the prediction slice is smaller than expected
                    if prediction_slice.shape[0] < actual_h or prediction_slice.shape[1] < actual_w:
                        actual_h = min(actual_h, prediction_slice.shape[0])
                        actual_w = min(actual_w, prediction_slice.shape[1])
                    
                    dem_30m_data[start_y_30m:start_y_30m+actual_h, start_x_30m:start_x_30m+actual_w] = \
                        prediction_slice[:actual_h, :actual_w]
            
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
    
    # Use the chunk-based inference with buffer for edge effect reduction
    dem_30m_cnn = chunk_based_inference(
        model, 
        dem_90m, 
        batch_size=batch_size,
        chunk_size=128,
        buffer_size=8  # 8-pixel buffer on each side to reduce edge effects
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