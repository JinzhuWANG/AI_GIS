import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from tqdm.auto import tqdm


np.random.seed(42) # For reproducibility



# #############################################################
# Previous results
# #############################################################
data = pd.read_csv('data/circle_data.csv')
data[['t1','t2']] = [0, 1]
data = pd.concat([
    data[['x1','y1','t1']].rename(columns={'x1':'x','y1':'y','t1':'t'}),
    data[['x2','y2','t2']].rename(columns={'x2':'x','y2':'y','t2':'t'})
])

# data.to_csv('data/circle_data_restructured.csv', index=False)


decision_line = pd.read_csv('data/circle_data_decision_line.csv')

sns.scatterplot(data=data, x='x', y='y', hue='t')
sns.lineplot(data=decision_line, x='x', y='y', color='green', sort=False, label='Decision Line')




# #############################################################
# Machine learning for linear regression
# #############################################################
# Extract features (x, y coordinates) and labels from the restructured data
X = torch.tensor(data[['x', 'y']].values, dtype=torch.float32)
y = torch.tensor(data['t'].values, dtype=torch.float32)

print(f"Dataset size: {len(X)} samples")
print(f"Input features: {X.shape[1]} (x, y coordinates)")
print(f"Class distribution: {int(y.sum())} red points, {int(len(y) - y.sum())} blue points")

# Initialize 2-layer perceptron parameters using torch.rand with requires_grad=True
# Layer 1: Input (2) -> Hidden (8)
W1 = torch.tensor(np.random.rand(2, 8) - 0.5, dtype=torch.float32, requires_grad=True)
b1 = torch.tensor(np.random.rand(8) - 0.5, dtype=torch.float32, requires_grad=True)

# Layer 2: Hidden (8) -> Output (1)
W2 = torch.tensor(np.random.rand(8, 1) - 0.5, dtype=torch.float32, requires_grad=True)
b2 = torch.tensor(np.random.rand(1) - 0.5, dtype=torch.float32, requires_grad=True)

print(f"\nNetwork architecture:")
print(f"Layer 1: {W1.shape} weights + {b1.shape} biases")
print(f"Layer 2: {W2.shape} weights + {b2.shape} biases")
print(f"Total parameters: {W1.numel() + b1.numel() + W2.numel() + b2.numel()}")

# Training parameters
learning_rate = 0.01
epochs = 30000

# Create optimizer
optimizer = optim.SGD([W1, b1, W2, b2], lr=learning_rate)

# Create loss criterion
criterion = nn.MSELoss()

# Training loop
for epoch in tqdm(range(epochs)):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    
    # Forward pass
    z1 = X @ W1 + b1  # Shape: [N, 8]
    a1 = torch.relu(z1)            # Shape: [N, 8]
    
    # Layer 2: Linear transformation + Sigmoid activation
    z2 = a1 @ W2 + b2 # Shape: [N, 1]
    y_pred = torch.sigmoid(z2).squeeze()  # Shape: [N]
    
    # Calculate loss
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()

# Final evaluation
with torch.no_grad():
    z1 = torch.matmul(X, W1) + b1
    a1 = torch.relu(z1)
    z2 = torch.matmul(a1, W2) + b2
    y_pred = torch.sigmoid(z2).squeeze()
    y_pred_binary = (y_pred > 0.5).float()
    final_accuracy = (y_pred_binary == y).float().mean()

print(f"\nFinal accuracy: {final_accuracy.item():.4f}")

# Generate decision boundary
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1

# Create a mesh grid
h = 0.1  # Step size
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Convert mesh to tensor
mesh_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Predict on mesh points
with torch.no_grad():
    z1_mesh = torch.matmul(mesh_points, W1) + b1
    a1_mesh = torch.relu(z1_mesh)
    z2_mesh = torch.matmul(a1_mesh, W2) + b2
    y_pred_mesh = torch.sigmoid(z2_mesh).squeeze()

# Reshape predictions to match mesh
Z = y_pred_mesh.numpy().reshape(xx.shape)



# Create a combined plot with the same extent
plt.figure(figsize=(12, 10))

contour_fill = plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
plt.colorbar(contour_fill, label='PyTorch Prediction Probability')
contour_line = plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')

sns.scatterplot(data=data, x='x', y='y', hue='t')
plt.plot(decision_line['x'], decision_line['y'], 'green', linewidth=3, label='Previous Decision Line')
plt.title(f'PyTorch 2-Layer Perceptron vs Previous Decision Line\nAccuracy: {final_accuracy.item():.3f}')

