import os
import xarray as xr
import numpy as np
from PIL import Image

# =============================================================================
# TRAINING DATA PROCESSING
# =============================================================================

# Load
train_coarse_data = xr.open_dataarray('data/images/train/coarse/train_coarse.nc', chunks='auto')

# Upsample 
train_upsampled_data = train_coarse_data.interp(
    height=range(256),
    width=range(256),
    method='linear',
    kwargs={"fill_value": "extrapolate"}
).astype(np.uint8)

# Save
os.makedirs('data/images/bilinear_interpolate/train', exist_ok=True)

train_upsampled_data.to_netcdf(
    'data/images/bilinear_interpolate/train/upsampled_bilinear.nc',
    engine='netcdf4',
    encoding={
        'data': {
            'dtype': 'uint8',
            'zlib': True,
            'complevel': 6
        }
    }
)

# Save training images as PNG files
print(f"Saving {len(train_upsampled_data)} training images...")
for i, train_image_data in enumerate(train_upsampled_data.compute()):
    train_image_pil = Image.fromarray(train_image_data.values.astype(np.uint8))
    train_image_pil.save(f'data/images/bilinear_interpolate/train/image_{i:04d}.png')



# =============================================================================
# TEST DATA PROCESSING
# =============================================================================

# Load 
test_coarse_data = xr.open_dataarray('data/images/test/coarse/test_coarse.nc', chunks='auto')
print(f"Test data loaded with shape: {test_coarse_data.shape}")

# Upsample
test_upsampled_data = test_coarse_data.interp(
    height=range(256),
    width=range(256),
    method='linear',
    kwargs={"fill_value": "extrapolate"}
).astype(np.uint8)

# Save
os.makedirs('data/images/bilinear_interpolate/test', exist_ok=True)

test_upsampled_data.to_netcdf(
    'data/images/bilinear_interpolate/test/upsampled_bilinear.nc',
    engine='netcdf4',
    encoding={
        'data': {
            'dtype': 'uint8',
            'zlib': True,
            'complevel': 6
        }
    }
)

print(f"Saving {len(test_upsampled_data)} test images...")
for i, test_image_data in enumerate(test_upsampled_data.compute()):
    test_image_pil = Image.fromarray(test_image_data.values.astype(np.uint8))
    test_image_pil.save(f'data/images/bilinear_interpolate/test/image_{i:04d}.png')


