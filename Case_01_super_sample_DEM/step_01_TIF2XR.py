import xarray as xr
import rioxarray as rxr
import numpy as np
from affine import Affine

# Read DEM tif
dem_tif = rxr.open_rasterio("data/DEM_30m.tif", chunks="auto")\
    .squeeze()\
    .drop_vars("band")\
    .astype(np.uint16)
    
    
# For better performance with large DEMs, specify chunk sizes
chunk_size = 256  # Adjust based on your DEM size
dem_tif.name = 'data'

dem_tif.to_netcdf(
    "data/DEM_30m.nc",
    engine='netcdf4',  # Move engine here
    encoding={
        'data': {
            'dtype': 'uint16',
            'zlib': True,
            'complevel': 6,
            'chunksizes': (chunk_size, chunk_size)  
        }
    }
)



# Downsample to 100m
dem_data = xr.open_dataarray('data/DEM_30m.nc', chunks=256)
dem_data_90m = dem_data.coarsen(x=3, y=3, boundary='trim').mean()

# Update transform
old_transform = dem_data.rio.transform()
new_transform = Affine(
    old_transform.a * 3, old_transform.b, old_transform.c,
    old_transform.d, old_transform.e * 3, old_transform.f
)
dem_data_90m.rio.write_transform(new_transform, inplace=True)

dem_data_90m.astype(np.uint16).to_netcdf(
    'data/DEM_100m.nc',
     engine='netcdf4',  # Move engine here
    encoding={
        'data': {
            'dtype': 'uint16',
            'zlib': True,
            'complevel': 6,
            'chunksizes': (chunk_size, chunk_size)  
        }
    })