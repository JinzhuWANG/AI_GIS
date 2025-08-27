import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xarray as xr

class ShapeDatasetGenerator:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.center = image_size // 2
        
        # Define consistent sizes for each shape (radius/half-width)
        self.circle_radius = 12  # Increased slightly to account for outline
        self.triangle_size = 14  # Distance from center to vertex
        self.rectangle_width = 18
        self.rectangle_height = 14
    
    def _draw_thick_line(self, img, x1, y1, x2, y2, thickness=2):
        """Draw a thick line on the image using multiple parallel lines"""
        # Convert to integers
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        
        # Basic line drawing using Bresenham's algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Determine line direction
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        # Main line points
        err = dx - dy
        x, y = x1, y1
        
        points = []
        while True:
            points.append((x, y))
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        # Draw thick line by adding thickness around each point
        half_thick = thickness // 2
        for px, py in points:
            for offset_x in range(-half_thick, half_thick + 1):
                for offset_y in range(-half_thick, half_thick + 1):
                    new_x, new_y = px + offset_x, py + offset_y
                    if 0 <= new_x < self.image_size and 0 <= new_y < self.image_size:
                        img[new_y, new_x] = 255
    
    def create_circle(self, rotation=0):
        """Create a binary image with a centered circle outline (2px width)"""
        # Create coordinate grids
        y, x = np.ogrid[:self.image_size, :self.image_size]
        
        # Calculate distance from center
        dist_from_center = np.sqrt((x - self.center)**2 + (y - self.center)**2)
        
        # Create circle outline with thicker width (3px radially = ~2px visually)
        outer_circle = dist_from_center <= self.circle_radius
        inner_circle = dist_from_center <= (self.circle_radius - 3)  # Increased from 2 to 3
        circle_outline = outer_circle & ~inner_circle
        
        return circle_outline.astype(np.uint8) * 255
    
    def create_triangle(self, rotation=0):
        """Create a binary image with a centered, rotated equilateral triangle outline (2px width)"""
        # Define equilateral triangle vertices (pointing up initially)
        angle_offset = np.pi / 2  # Start pointing up
        vertices = []
        for i in range(3):
            angle = angle_offset + i * 2 * np.pi / 3 + np.radians(rotation)
            x = self.center + self.triangle_size * np.cos(angle)
            y = self.center + self.triangle_size * np.sin(angle)
            vertices.append([x, y])
        
        # Create image using line drawing approach
        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Draw triangle edges with 2px thickness
        for i in range(3):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % 3]
            
            # Use thick line drawing
            self._draw_thick_line(img, x1, y1, x2, y2, thickness=2)
        
        return img
    
    def create_rectangle(self, rotation=0):
        """Create a binary image with a centered, rotated rectangle outline (2px width)"""
        # Define rectangle corners before rotation
        half_w, half_h = self.rectangle_width // 2, self.rectangle_height // 2
        corners = np.array([
            [-half_w, -half_h],
            [half_w, -half_h],
            [half_w, half_h],
            [-half_w, half_h]
        ], dtype=float)
        
        # Apply rotation
        rotation_rad = np.radians(rotation)
        cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
        rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
        
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners += self.center  # Translate to center
        
        # Create image using line drawing approach
        img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Draw rectangle edges with 2px thickness
        for i in range(4):
            x1, y1 = rotated_corners[i]
            x2, y2 = rotated_corners[(i + 1) % 4]
            
            # Draw thick line between consecutive corners
            self._draw_thick_line(img, x1, y1, x2, y2, thickness=2)
        
        return img
    
    def generate_dataset(self, num_samples_per_class=1000, save_path="shapes_dataset", save_xarray=True):
        """Generate the complete dataset"""
        shapes = {
            'circle': self.create_circle,
            'triangle': self.create_triangle,
            'rectangle': self.create_rectangle
        }
        
        # Create directories
        os.makedirs(save_path, exist_ok=True)
        for shape_name in shapes.keys():
            os.makedirs(f"{save_path}/{shape_name}", exist_ok=True)
        
        dataset = []
        labels = []
        
        # For xarray storage
        shape_datasets = {}
        
        for label, (shape_name, shape_func) in enumerate(shapes.items()):
            print(f"Generating {num_samples_per_class} {shape_name} samples...")
            
            # Generate unique rotations without replacement
            if num_samples_per_class <= 360:
                # If we need fewer samples than 360, sample without replacement
                rotations = np.random.choice(360, size=num_samples_per_class, replace=False)
            else:
                # If we need more than 360 samples, we'll have to allow some repeats
                # but we'll try to minimize them
                base_rotations = np.arange(360)
                extra_needed = num_samples_per_class - 360
                extra_rotations = np.random.choice(360, size=extra_needed, replace=True)
                rotations = np.concatenate([base_rotations, extra_rotations])
                np.random.shuffle(rotations)  # Shuffle to mix base and extra rotations
            
            # Store all images for this shape
            shape_images = []
            
            for i, rotation in enumerate(rotations):
                # Generate shape image
                img = shape_func(int(rotation))
                
                # Save PNG image
                img_pil = Image.fromarray(img, mode='L')
                img_pil.save(f"{save_path}/{shape_name}/{shape_name}_{i:04d}_rot{rotation}.png")
                
                # Add to dataset arrays
                dataset.append(img)
                labels.append(label)
                shape_images.append(img)
            
            # Create xarray dataset for this shape
            if save_xarray:
                shape_array = np.array(shape_images)  # Shape: (num_samples, height, width)
                
                # Create xarray DataArray
                da = xr.DataArray(
                    shape_array,
                    dims=['rotation', 'height', 'width'],
                    coords={
                        'rotation': rotations,
                        'height': range(self.image_size),
                        'width': range(self.image_size)
                    },
                    name='data',  # Use 'data' as the name
                    attrs={
                        'shape_type': shape_name,
                        'image_size': self.image_size,
                        'outline_thickness': '2px',
                        'description': f'{shape_name} shapes with random rotations'
                    }
                )
                
                # Save individual shape xarray to file with encoding
                encoding = {
                    'data': {  # Now matches your original encoding format
                        'dtype': 'uint8',
                        'zlib': True,
                        'complevel': 6,
                    }
                }
                da.to_netcdf(f"{save_path}/{shape_name}_dataset.nc", encoding=encoding)
                print(f"Saved {shape_name} xarray dataset to {save_path}/{shape_name}_dataset.nc")
        
        return np.array(dataset), np.array(labels)
    
    def visualize_samples(self, num_samples=3):
        """Visualize sample images from each class"""
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples*2, 6))
        
        shapes = [
            ('Circle', self.create_circle),
            ('Triangle', self.create_triangle),
            ('Rectangle', self.create_rectangle)
        ]
        
        for row, (shape_name, shape_func) in enumerate(shapes):
            for col in range(num_samples):
                rotation = np.random.randint(0, 360)
                img = shape_func(rotation)
                
                axes[row, col].imshow(img, cmap='gray')
                axes[row, col].set_title(f"{shape_name}\nRotation: {rotation}°")
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create generator
    generator = ShapeDatasetGenerator(image_size=32)
    
    # Visualize some samples
    print("Visualizing sample shapes...")
    generator.visualize_samples(num_samples=4)
    
    # Generate full dataset
    print("Generating dataset...")
    X, y = generator.generate_dataset(num_samples_per_class=100, save_path='data')  # Reduced for demo
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")  # 0=circle, 1=triangle, 2=rectangle
    
    # Display class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['Circle', 'Triangle', 'Rectangle']
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"{class_names[label]}: {count} samples")