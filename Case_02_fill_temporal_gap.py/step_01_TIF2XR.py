import xarray as xr
import rioxarray as rxr
import numpy as np


# Get Leaf Area Index (LAI)
lai = rxr.open_rasterio("data/LAI_20240731_20250731.tif")
lai['band'] = [f'LAI_{i[:-4]}' for i in lai.attrs['long_name']]


chunk_size = 256  # Adjust based on your DEM size
lai.name = 'data'

lai.to_netcdf(
    "data/LAI_20240731_20250731.nc",
    engine='netcdf4',  # Move engine here
    encoding={
        'data': {
            'dtype': 'uint16',
            'zlib': True,
            'complevel': 6,
            'chunksizes': (1, chunk_size, chunk_size)  
        }
    }
)
