import xarray as xr
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def calculate_rgb_metrics(y_true, y_pred, method_name):
    """Calculate performance metrics between true and predicted RGB values"""
    # Flatten arrays for calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove any NaN values
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
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
    
    # Peak Signal-to-Noise Ratio (PSNR) for images
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    
    # Structural Similarity Index (simplified version)
    mean_true = np.mean(y_true_clean)
    mean_pred = np.mean(y_pred_clean)
    var_true = np.var(y_true_clean)
    var_pred = np.var(y_pred_clean)
    covar = np.mean((y_true_clean - mean_true) * (y_pred_clean - mean_pred))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ssim = ((2 * mean_true * mean_pred + c1) * (2 * covar + c2)) / \
           ((mean_true**2 + mean_pred**2 + c1) * (var_true + var_pred + c2))
    
    metrics = {
        'Method': method_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Correlation': correlation,
        'P-value': p_value,
        'Bias': bias,
        'PSNR': psnr,
        'SSIM': ssim,
        'N_points': len(y_true_clean)
    }
    
    return metrics

def calculate_channel_metrics(rgb_true, rgb_pred, method_name):
    """Calculate metrics for each RGB channel separately"""
    channel_names = ['Red', 'Green', 'Blue']
    channel_metrics = []
    
    for i, channel in enumerate(channel_names):
        channel_true = rgb_true[:, :, :, i]  # (samples, height, width)
        channel_pred = rgb_pred[:, :, :, i]
        
        metrics = calculate_rgb_metrics(channel_true, channel_pred, f"{method_name}_{channel}")
        channel_metrics.append(metrics)
    
    return channel_metrics

def random_sample_rgb_images(rgb_arrays, n_samples=5, seed=42):
    """Randomly sample n images from RGB arrays for visualization"""
    np.random.seed(seed)
    
    # Get the number of samples available
    ref_array = list(rgb_arrays.values())[0]
    n_total_samples = ref_array.shape[0]
    
    if n_samples > n_total_samples:
        print(f"Warning: Requested {n_samples} samples but only {n_total_samples} available. Using all samples.")
        sample_indices = range(n_total_samples)
    else:
        sample_indices = np.random.choice(n_total_samples, size=n_samples, replace=False)
    
    # Extract sampled images
    sampled_images = {}
    for name, array in rgb_arrays.items():
        sampled_images[name] = array[sample_indices]
    
    print(f"Sampled {len(sample_indices)} random images for visualization")
    return sampled_images, sample_indices

def create_rgb_comparison_plot(sampled_images, sample_indices, save_path=None):
    """Create comparison plots for RGB images"""
    n_samples = len(sample_indices)
    methods = list(sampled_images.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(n_samples, n_methods, figsize=(4*n_methods, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for i, sample_idx in enumerate(sample_indices):
        for j, method in enumerate(methods):
            image = sampled_images[method][i]
            # Convert from (height, width, channels) to displayable format
            # Ensure values are in [0, 255] range for display
            if image.max() <= 1.0:
                image_display = (image * 255).astype(np.uint8)
            else:
                image_display = np.clip(image, 0, 255).astype(np.uint8)
            
            axes[i, j].imshow(image_display)
            axes[i, j].set_title(f'{method} - Sample {sample_idx}')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"RGB comparison plot saved to {save_path}")
    
    plt.show()

def predict_with_cnn_model(low_res_data, model_path):
    """Use trained CNN model to predict high-resolution images"""
    print(f"Loading CNN model from {model_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the traced model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    # Convert low-res data to PyTorch format
    # Input: (N, H, W, C) -> (N, C, H, W)
    low_res_torch = np.transpose(low_res_data, (0, 3, 1, 2))
    low_res_torch = low_res_torch.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    predictions = []
    batch_size = 8
    
    print(f"Predicting on {low_res_torch.shape[0]} images...")
    
    with torch.no_grad():
        for i in range(0, low_res_torch.shape[0], batch_size):
            batch = low_res_torch[i:i+batch_size]
            batch_tensor = torch.FloatTensor(batch).to(device)
            
            pred_batch = model(batch_tensor)
            pred_batch = pred_batch.cpu().numpy()
            
            predictions.append(pred_batch)
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Convert back to (N, H, W, C) format and scale to [0, 255]
    predictions = np.transpose(predictions, (0, 2, 3, 1))
    predictions = np.clip(predictions * 255, 0, 255).astype(np.uint8)
    
    print(f"CNN predictions completed. Output shape: {predictions.shape}")
    return predictions

if __name__ == "__main__":
    print("Step 4: Evaluating RGB image super-resolution results...")
    
    # Load datasets
    print("\n1. Loading RGB image datasets...")
    
    # Load original high-resolution images (256x256)
    high_res_original = xr.open_dataarray('data/test.nc')
    print(f"Original high-resolution images shape: {high_res_original.shape}")
    
    # Load low-resolution images (64x64)
    low_res_images = xr.open_dataarray('data/test_coarse2.nc')
    print(f"Low-resolution images shape: {low_res_images.shape}")
    
    # Load bilinear interpolated result (from step 2)
    high_res_bilinear = xr.open_dataarray('data/train_upsampled_bilinear.nc')
    print(f"Bilinear upsampled images shape: {high_res_bilinear.shape}")
    
    # Convert to numpy arrays
    high_res_original_np = high_res_original.values
    low_res_images_np = low_res_images.values
    high_res_bilinear_np = high_res_bilinear.values
    
    # Use test data for evaluation if available, otherwise use a subset of training data
    if high_res_original_np.shape[0] > 0:
        print(f"Using test data for evaluation: {high_res_original_np.shape[0]} samples")
        eval_high_res = high_res_original_np
        eval_low_res = low_res_images_np
        eval_bilinear = high_res_bilinear_np[:high_res_original_np.shape[0]]  # Match sample count
    else:
        print("No test data found, using subset of training data")
        # Use a subset for evaluation
        n_eval = min(20, high_res_bilinear_np.shape[0])
        eval_high_res = high_res_bilinear_np[:n_eval]  # Placeholder - in practice, use separate test set
        eval_low_res = low_res_images_np[:n_eval]
        eval_bilinear = high_res_bilinear_np[:n_eval]
    
    # Predict using CNN model if available
    cnn_model_path = 'data/RGB_CNN_super_resolution_traced_final.pt'
    if os.path.exists(cnn_model_path):
        print(f"\n2. Generating CNN predictions...")
        eval_cnn = predict_with_cnn_model(eval_low_res, cnn_model_path)
        
        # Ensure all arrays have the same shape for comparison
        min_samples = min(eval_high_res.shape[0], eval_bilinear.shape[0], eval_cnn.shape[0])
        eval_high_res = eval_high_res[:min_samples]
        eval_bilinear = eval_bilinear[:min_samples]
        eval_cnn = eval_cnn[:min_samples]
        
        print(f"Using {min_samples} samples for evaluation")
        
        # Calculate metrics
        print(f"\n3. Calculating performance metrics...")
        
        # Overall metrics
        bilinear_metrics = calculate_rgb_metrics(eval_high_res, eval_bilinear, "Bilinear")
        cnn_metrics = calculate_rgb_metrics(eval_high_res, eval_cnn, "CNN")
        
        # Channel-wise metrics
        bilinear_channel_metrics = calculate_channel_metrics(eval_high_res, eval_bilinear, "Bilinear")
        cnn_channel_metrics = calculate_channel_metrics(eval_high_res, eval_cnn, "CNN")
        
        # Combine all metrics
        all_metrics = [bilinear_metrics, cnn_metrics] + bilinear_channel_metrics + cnn_channel_metrics
        
    else:
        print(f"\nCNN model not found at {cnn_model_path}")
        print("Evaluating only bilinear interpolation...")
        
        # Calculate metrics for bilinear only
        bilinear_metrics = calculate_rgb_metrics(eval_high_res, eval_bilinear, "Bilinear")
        bilinear_channel_metrics = calculate_channel_metrics(eval_high_res, eval_bilinear, "Bilinear")
        
        all_metrics = [bilinear_metrics] + bilinear_channel_metrics
        eval_cnn = None
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_metrics)
    print(f"\n4. Performance Results:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv('data/RGB_evaluation_results.csv', index=False)
    print(f"\nResults saved to 'data/RGB_evaluation_results.csv'")
    
    # Create visualization
    print(f"\n5. Creating comparison visualizations...")
    
    # Prepare images for visualization
    viz_images = {
        'Original (256x256)': eval_high_res,
        'Bilinear (256x256)': eval_bilinear
    }
    
    if eval_cnn is not None:
        viz_images['CNN (256x256)'] = eval_cnn
    
    # Sample random images for visualization
    sampled_images, sample_indices = random_sample_rgb_images(viz_images, n_samples=3)
    
    # Create comparison plot
    create_rgb_comparison_plot(sampled_images, sample_indices, 'data/RGB_comparison_samples.png')
    
    # Create metrics comparison plot
    if eval_cnn is not None:
        methods = ['Bilinear', 'CNN']
        metrics = ['RMSE', 'MAE', 'PSNR', 'SSIM']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            bilinear_val = bilinear_metrics[metric]
            cnn_val = cnn_metrics[metric]
            
            bars = axes[i].bar(methods, [bilinear_val, cnn_val], 
                             color=['skyblue', 'lightcoral'], alpha=0.7)
            axes[i].set_ylabel(metric)
            axes[i].set_title(f'{metric} Comparison')
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, [bilinear_val, cnn_val]):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('data/RGB_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"\n6. Summary:")
    print("="*50)
    if eval_cnn is not None:
        print(f"Bilinear Interpolation PSNR: {bilinear_metrics['PSNR']:.2f} dB")
        print(f"CNN Super-Resolution PSNR: {cnn_metrics['PSNR']:.2f} dB")
        print(f"PSNR Improvement: {cnn_metrics['PSNR'] - bilinear_metrics['PSNR']:.2f} dB")
        print(f"")
        print(f"Bilinear SSIM: {bilinear_metrics['SSIM']:.4f}")
        print(f"CNN SSIM: {cnn_metrics['SSIM']:.4f}")
        print(f"SSIM Improvement: {cnn_metrics['SSIM'] - bilinear_metrics['SSIM']:.4f}")
    else:
        print(f"Bilinear Interpolation PSNR: {bilinear_metrics['PSNR']:.2f} dB")
        print(f"Bilinear SSIM: {bilinear_metrics['SSIM']:.4f}")
    
    print("\nEvaluation completed! Check the generated files:")
    print("- data/RGB_evaluation_results.csv")
    print("- data/RGB_comparison_samples.png")
    if eval_cnn is not None:
        print("- data/RGB_metrics_comparison.png")
