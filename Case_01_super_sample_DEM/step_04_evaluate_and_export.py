import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, method_name):
    """Calculate performance metrics between true and predicted values"""
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print(f"Warning: No valid data points for {method_name}")
        return {}
    
    # Calculate metrics
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # Pearson correlation
    correlation, p_value = pearsonr(y_true_clean, y_pred_clean)
    
    # Bias (mean error)
    bias = np.mean(y_pred_clean - y_true_clean)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / y_true_clean)) * 100
    
    metrics = {
        'Method': method_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Correlation': correlation,
        'P-value': p_value,
        'Bias': bias,
        'MAPE (%)': mape,
        'N_points': len(y_true_clean)
    }
    
    return metrics

def save_to_compressed_tif(data_array, filename, dtype=np.int16, compression='lzw'):
    """Save xarray DataArray to compressed GeoTIFF using rioxarray"""
    print(f"Saving {filename} as compressed GeoTIFF...")
    
    # Convert to the specified dtype
    if dtype == np.int16:
        # For int16, we need to handle the range properly
        data_values = data_array.values.astype(np.float32)
        # Clip to int16 range and convert
        data_values = np.clip(data_values, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
        data_array = data_array.copy()
        data_array.values = data_values.astype(np.int16)
    else:
        data_array = data_array.astype(dtype)
    
    # Ensure rioxarray is available
    if not hasattr(data_array, 'rio'):
        import rioxarray as rio
        data_array = data_array.rio.write_crs("EPSG:4326")  # Default CRS if none exists
    
    # Save using rioxarray with compression
    data_array.rio.to_raster(
        filename,
        compress=compression,
        tiled=True,
        blockxsize=256,
        blockysize=256
    )
    
    print(f"Saved {filename} ({data_array.shape}) as compressed GeoTIFF with {compression} compression")

def random_sample_points(data_arrays, n_points=10000, seed=42):
    """Randomly sample n_points from multiple aligned data arrays"""
    np.random.seed(seed)
    
    # Get the shape of the arrays (assume all have same shape)
    ref_array = list(data_arrays.values())[0]
    height, width = ref_array.shape
    
    # Generate random indices
    total_pixels = height * width
    if n_points > total_pixels:
        print(f"Warning: Requested {n_points} points but only {total_pixels} available. Using all points.")
        # Use all points
        y_indices, x_indices = np.meshgrid(range(height), range(width), indexing='ij')
        y_indices = y_indices.flatten()
        x_indices = x_indices.flatten()
    else:
        # Random sampling
        random_indices = np.random.choice(total_pixels, size=n_points, replace=False)
        y_indices, x_indices = np.unravel_index(random_indices, (height, width))
    
    # Extract values from all arrays
    sampled_data = {}
    for name, array in data_arrays.items():
        values = array.values[y_indices, x_indices]
        sampled_data[name] = values
    
    # Also return coordinates for reference
    if hasattr(ref_array, 'x') and hasattr(ref_array, 'y'):
        x_coords = ref_array.x.values[x_indices]
        y_coords = ref_array.y.values[y_indices]
        sampled_data['x_coords'] = x_coords
        sampled_data['y_coords'] = y_coords
    
    print(f"Sampled {len(y_indices)} random points from {height}x{width} arrays")
    return sampled_data

def create_scatter_plots(sampled_data, original_key, methods, save_path=None):
    """Create scatter plots comparing methods against original data"""
    n_methods = len(methods)
    
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]
    
    original_values = sampled_data[original_key]
    
    for i, method in enumerate(methods):
        predicted_values = sampled_data[method]
        
        # Create scatter plot
        axes[i].scatter(original_values, predicted_values, alpha=0.5, s=1)
        
        # Add 1:1 line
        min_val = min(np.min(original_values), np.min(predicted_values))
        max_val = max(np.max(original_values), np.max(predicted_values))
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate R²
        r2 = r2_score(original_values, predicted_values)
        
        axes[i].set_xlabel('Original 30m DEM')
        axes[i].set_ylabel(f'{method} 30m DEM')
        axes[i].set_title(f'{method} vs Original (R² = {r2:.4f})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plots saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Step 4: Evaluating and exporting results...")
    
    # Load all datasets
    print("\n1. Loading datasets...")
    
    # Load original 30m DEM
    dem_30m_original = xr.open_dataarray('data/DEM_30m.nc')
    print(f"Original 30m DEM shape: {dem_30m_original.shape}")
    
    # Load bilinear interpolated result (from step 2)
    dem_30m_interpolated = xr.open_dataarray('data/DEM_30m_interpolated.nc')
    print(f"Bilinear interpolated 30m DEM shape: {dem_30m_interpolated.shape}")
    
    # Load CNN prediction result (from step 3-2)
    dem_30m_cnn = xr.open_dataarray('data/DEM_30m_CNN_prediction.nc')
    print(f"CNN predicted 30m DEM shape: {dem_30m_cnn.shape}")
    
    # Check if shapes match
    if not (dem_30m_original.shape == dem_30m_interpolated.shape == dem_30m_cnn.shape):
        print("Warning: Dataset shapes don't match!")
        print(f"Original: {dem_30m_original.shape}")
        print(f"Interpolated: {dem_30m_interpolated.shape}")
        print(f"CNN: {dem_30m_cnn.shape}")
        
        # Find the common shape (smallest)
        min_height = min(dem_30m_original.shape[0], dem_30m_interpolated.shape[0], dem_30m_cnn.shape[0])
        min_width = min(dem_30m_original.shape[1], dem_30m_interpolated.shape[1], dem_30m_cnn.shape[1])
        
        # Crop all to the same size
        dem_30m_original = dem_30m_original.isel(y=slice(0, min_height), x=slice(0, min_width))
        dem_30m_interpolated = dem_30m_interpolated.isel(y=slice(0, min_height), x=slice(0, min_width))
        dem_30m_cnn = dem_30m_cnn.isel(y=slice(0, min_height), x=slice(0, min_width))
        
        print(f"Cropped all datasets to {min_height} x {min_width}")
    
    # Save to compressed TIFFs
    print("\n2. Saving datasets as compressed GeoTIFFs...")
    
    save_to_compressed_tif(dem_30m_original, 'data/DEM_30m_original.tif', dtype=np.int16)
    save_to_compressed_tif(dem_30m_interpolated, 'data/DEM_30m_interpolated.tif', dtype=np.int16)
    save_to_compressed_tif(dem_30m_cnn, 'data/DEM_30m_CNN_prediction.tif', dtype=np.int16)
    
    # Random sampling for evaluation
    print("\n3. Randomly sampling 10,000 points for evaluation...")
    
    data_arrays = {
        'Original': dem_30m_original,
        'Bilinear': dem_30m_interpolated,
        'CNN': dem_30m_cnn
    }
    
    sampled_data = random_sample_points(data_arrays, n_points=10000, seed=42)
    
    # Calculate performance metrics
    print("\n4. Calculating performance metrics...")
    
    original_values = sampled_data['Original']
    methods = ['Bilinear', 'CNN']
    
    results = []
    
    for method in methods:
        predicted_values = sampled_data[method]
        metrics = calculate_metrics(original_values, predicted_values, method)
        results.append(metrics)
        
        print(f"\n{method} Performance Metrics:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  R²: {metrics['R²']:.4f}")
        print(f"  Correlation: {metrics['Correlation']:.4f}")
        print(f"  Bias: {metrics['Bias']:.4f}")
        print(f"  MAPE: {metrics['MAPE (%)']:.2f}%")
    
    # Create results DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/performance_metrics.csv', index=False)
    print(f"\nPerformance metrics saved to 'data/performance_metrics.csv'")
    
    # Display comparison table
    print("\n5. Performance Comparison Summary:")
    print("="*80)
    print(f"{'Metric':<15} {'Bilinear':<15} {'CNN':<15} {'CNN Better?':<15}")
    print("="*80)
    
    bilinear_metrics = results[0]
    cnn_metrics = results[1]
    
    for metric in ['RMSE', 'MAE', 'R²', 'Correlation', 'MAPE (%)']:
        bilinear_val = bilinear_metrics[metric]
        cnn_val = cnn_metrics[metric]
        
        # For R² and Correlation, higher is better; for others, lower is better
        if metric in ['R²', 'Correlation']:
            better = "Yes" if cnn_val > bilinear_val else "No"
        else:
            better = "Yes" if cnn_val < bilinear_val else "No"
        
        print(f"{metric:<15} {bilinear_val:<15.4f} {cnn_val:<15.4f} {better:<15}")
    
    print("="*80)
    
    # Create scatter plots
    print("\n6. Creating scatter plots...")
    create_scatter_plots(
        sampled_data, 
        'Original', 
        methods, 
        save_path='data/scatter_comparison.png'
    )
    
    # Save sampled data for further analysis if needed
    sampled_df = pd.DataFrame(sampled_data)
    sampled_df.to_csv('data/sampled_evaluation_points.csv', index=False)
    print(f"Sampled evaluation points saved to 'data/sampled_evaluation_points.csv'")
    
    print("\nStep 4 completed successfully!")
    print("\nFiles created:")
    print("  - data/DEM_30m_original.tif")
    print("  - data/DEM_30m_interpolated.tif") 
    print("  - data/DEM_30m_CNN_prediction.tif")
    print("  - data/performance_metrics.csv")
    print("  - data/scatter_comparison.png")
    print("  - data/sampled_evaluation_points.csv")