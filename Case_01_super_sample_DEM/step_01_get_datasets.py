import numpy as np
import xarray as xr

from sklearn.model_selection import train_test_split
from datasets import load_dataset

train_size, test_size = 0.7, 0.3


# Get data
# https://huggingface.co/datasets/blanchon/UC_Merced
UC_Merced = load_dataset(
    "blanchon/UC_Merced", 
    cache_dir="data/UC_Merced"
)

# Split the dataset into train and test sets
UC_Merced_train_test = UC_Merced["train"].train_test_split(
    test_size=test_size,
    seed=42  # Set a seed for reproducibility
)

train_data = UC_Merced_train_test["train"]
test_data = UC_Merced_train_test["test"]

# Get the first image and convert it to numpy array
from PIL import Image

# First ensure all images have the same shape (256x256x3)
train_images = []
for img in train_data['image']:
    img_array = np.array(img.resize((256, 256)))
    train_images.append(img_array)
    
test_images = []
for img in test_data['image']:
    img_array = np.array(img.resize((256, 256)))
    test_images.append(img_array)

train_xr = xr.DataArray(
    np.stack(train_images),
    dims=['sample', "height", "width", "channels"],
    coords={
        "sample": range(len(train_data['image'])),
        "height": range(256),
        "width": range(256),
        "channels": ["r", "g", "b"]
    }
)

test_xr = xr.DataArray(
    np.stack(test_images),
    dims=['sample', "height", "width", "channels"],
    coords={
        "sample": range(len(test_data['image'])),
        "height": range(256),
        "width": range(256),
        "channels": ["r", "g", "b"]
    }
)

train_xr_coarse2 = train_xr.coarsen
