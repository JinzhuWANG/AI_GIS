
import numpy as np
import xarray as xr

from step_01_create_data import ShapeDatasetGenerator
from matplotlib import pyplot as plt

shape_gen = ShapeDatasetGenerator()


# Load shapes
circle = xr.open_dataarray("data/circle_dataset.nc")
triangle = xr.open_dataarray("data/triangle_dataset.nc")
rectangle = xr.open_dataarray("data/rectangle_dataset.nc")



sample = triangle[42].values
plt.imshow(sample, cmap='gray')




# Circle test
covariance_circle = []
for rotation in range(361):
    covariance = (shape_gen.create_circle(rotation) * sample).sum()
    covariance_circle.append(covariance)

max(covariance_circle)



# Triangle test
covariance_triangle = []
for rotation in range(361):
    covariance = (shape_gen.create_triangle(rotation) * sample).sum()
    covariance_triangle.append(covariance)

max(covariance_triangle)



# Rectangle test
covariance_rectangle = []
for rotation in range(361):
    covariance = (shape_gen.create_rectangle(rotation) * sample).sum()
    covariance_rectangle.append(covariance)

max(covariance_rectangle)
