import xarray as xr
import numpy as np


# Load data
low_res_images = xr.open_dataarray('data/train_coarse2.nc', chunks='auto')

# Perform bilinear interpolation to upsample 64x64 images to 256x256
upsampled_images = low_res_images.interp(
    height=range(256),
    width=range(256),
    method='linear',
    kwargs={"fill_value": "extrapolate"}
).astype(np.uint8)


# Save the interpolated result
upsampled_images.to_netcdf(
    'data/train_upsampled_bilinear.nc',
    engine='netcdf4',
    encoding={
        'data': {
            'dtype': 'uint8',
            'zlib': True,
            'complevel': 6
        }
    }
)

