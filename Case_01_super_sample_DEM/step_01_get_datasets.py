from matplotlib import pyplot as plt
import numpy as np
import xarray as xr

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


# Save to disk
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

train_xr.to_netcdf('data/train.nc', encoding=encoding)
test_xr.to_netcdf('data/test.nc', encoding=encoding)
train_xr_coarse2.to_netcdf('data/train_coarse2.nc', encoding=encoding)
test_xr_coarse2.to_netcdf('data/test_coarse2.nc', encoding=encoding)
