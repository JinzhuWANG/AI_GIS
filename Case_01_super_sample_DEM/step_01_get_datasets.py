import os
import numpy as np
import xarray as xr

from PIL import Image

from datasets import load_dataset
from sklearn.model_selection import train_test_split


train_size, test_size = 0.7, 0.3


# Get data
# https://huggingface.co/datasets/blanchon/UC_Merced
UC_Merced = load_dataset(
    "blanchon/UC_Merced", 
    cache_dir="data/UC_Merced"
)


# First ensure all images have the same shape (256x256x3)
all_images = []
for img in UC_Merced['train']['image']:
    img_array = np.array(img.resize((256, 256)))
    all_images.append(img_array)

train_data, test_data = train_test_split(all_images, test_size=test_size, random_state=42)


train_xr = xr.DataArray(
    np.stack(train_data),
    dims=['sample', "height", "width", "channels"],
    coords={
        "sample": range(len(train_data)),
        "height": range(256),
        "width": range(256),
        "channels": ["r", "g", "b"]
    }
)

test_xr = xr.DataArray(
    np.stack(test_data),
    dims=['sample', "height", "width", "channels"],
    coords={
        "sample": range(len(test_data)),
        "height": range(256),
        "width": range(256),
        "channels": ["r", "g", "b"]
    }
)

train_xr_coarse2 = train_xr.coarsen(height=4, width=4, boundary='trim').mean().astype(np.uint8)
test_xr_coarse2 = test_xr.coarsen(height=4, width=4, boundary='trim').mean().astype(np.uint8)


# Save to disk as PNG files
os.makedirs('data/images/train/original', exist_ok=True)
os.makedirs('data/images/train/coarse', exist_ok=True)
os.makedirs('data/images/test/original', exist_ok=True)
os.makedirs('data/images/test/coarse', exist_ok=True)

for i, img_array in enumerate(train_data):
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    img_pil.save(f'data/images/train/original/image_{i:04d}.png')
    coarse_img = train_xr_coarse2[i].values.astype(np.uint8)
    coarse_pil = Image.fromarray(coarse_img)
    coarse_pil.save(f'data/images/train/coarse/image_{i:04d}.png')

for i, img_array in enumerate(test_data):
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    img_pil.save(f'data/images/test/original/image_{i:04d}.png')
    coarse_img = test_xr_coarse2[i].values.astype(np.uint8)
    coarse_pil = Image.fromarray(coarse_img)
    coarse_pil.save(f'data/images/test/coarse/image_{i:04d}.png')



# Save NetCDF files to corresponding directories
encoding={
    'data': {
        'dtype': 'uint8',
        'zlib': True,
        'complevel': 6,
    }
}

train_xr.name = 'data'
test_xr.name = 'data'
train_xr_coarse2.name = 'data'
test_xr_coarse2.name = 'data'

train_xr.to_netcdf('data/images/train/original/train.nc', encoding=encoding)
test_xr.to_netcdf('data/images/test/original/test.nc', encoding=encoding)
train_xr_coarse2.to_netcdf('data/images/train/coarse/train_coarse2.nc', encoding=encoding)
test_xr_coarse2.to_netcdf('data/images/test/coarse/test_coarse2.nc', encoding=encoding)

