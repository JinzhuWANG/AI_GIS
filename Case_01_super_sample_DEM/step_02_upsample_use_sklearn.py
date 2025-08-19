import xarray as xr
import numpy as np


# Load the original 30m DEM to get the target grid
dem_30m_original = xr.open_dataarray('data/DEM_30m.nc', chunks=256)

# Load the 90m DEM data 
dem_90m = xr.open_dataarray('data/DEM_90m.nc', chunks=256)

# Perform bilinear interpolation to upsample 90m data to 30m grid
dem_30m_interpolated = dem_90m.interp(
    x=dem_30m_original.x,
    y=dem_30m_original.y,
    method='linear'  # This is bilinear interpolation for 2D data
)


# Save the interpolated result
dem_30m_interpolated.astype(np.uint16).to_netcdf(
    'data/DEM_30m_interpolated.nc',
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


