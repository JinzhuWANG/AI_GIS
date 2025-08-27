import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from step_01_create_data import ShapeDatasetGenerator


# Create shapes
shape_gen = ShapeDatasetGenerator()

circle = shape_gen.create_circle()
triangle = shape_gen.create_triangle()
rectangle = shape_gen.create_rectangle()


# plot the data
plt.figure(figsize=(9, 3))
fig, ax = plt.subplots(1, 3)
ax[0].imshow(circle, cmap='gray')
ax[0].set_title("Circle")
ax[1].imshow(triangle, cmap='gray')
ax[1].set_title("Triangle")
ax[2].imshow(rectangle, cmap='gray')
ax[2].set_title("Rectangle")


# Multiply circle to other shapes
circle_circle = (circle * circle).sum()
circle_triangle = (circle * triangle).sum()
circle_rectangle = (circle * rectangle).sum()

# Multiply triangle to other shapes
triangle_triangle = (triangle * triangle).sum()
triangle_circle = (triangle * circle).sum()
triangle_rectangle = (triangle * rectangle).sum()

# Multiply rectangle to other shapes
rectangle_rectangle = (rectangle * rectangle).sum()
rectangle_circle = (rectangle * circle).sum()
rectangle_triangle = (rectangle * triangle).sum()


