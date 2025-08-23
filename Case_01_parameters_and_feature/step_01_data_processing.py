import numpy as np
import pandas as pd
import seaborn as sns

# ###################################################################
#   Height Weight Data
# ###################################################################

height_weight = pd.read_csv('data/height-weight.csv')

height_weight_20 = height_weight.sample(n=20, random_state=42)
height_weight_20.to_csv('data/height-weight-20.csv', index=False)



# ###################################################################
#   Circle data
# ###################################################################

def generate_circle_data(center_x=10, center_y=10, radius=8, jitter_range=(-1, 1),
                        n_points=20, random_seed=42):

    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate independent angles and jitter
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    jitter = np.random.uniform(jitter_range[0], jitter_range[1], n_points)
    
    # Calculate x, y coordinates
    x = center_x + (radius + jitter) * np.cos(angles)
    y = center_y + (radius + jitter) * np.sin(angles)
    
    # Create DataFrame
    circle_data = pd.DataFrame({
        'x': x,
        'y': y
    })
    
    return circle_data

# Generate the two circles with different parameters
circle_1 = generate_circle_data(center_x=10, center_y=10, radius=8, jitter_range=(-2, 2), 
                               n_points=30, random_seed=42)
circle_2 = generate_circle_data(center_x=10, center_y=10, radius=4, jitter_range=(-2, 2), 
                               n_points=30, random_seed=123)  

# Combine the data
circle_data = pd.DataFrame({
    'x1': circle_1['x'],
    'y1': circle_1['y'],
    'x2': circle_2['x'],
    'y2': circle_2['y']
})

# Visualize the data
sns.scatterplot(data=circle_data, x='x1', y='y1', color='blue', label='Circle 1')
sns.scatterplot(data=circle_data, x='x2', y='y2', color='red', label='Circle 2')

# Save to CSV file
circle_data.to_csv('data/circle_data.csv', index=False)


