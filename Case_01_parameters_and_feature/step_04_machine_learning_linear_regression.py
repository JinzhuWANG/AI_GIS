import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# Read the dataset
data = pd.read_csv('data/height-weight-20.csv')
sns.scatterplot(x=data['Height'], y=data['Weight'])


# #############################################################
# Previous results
# #############################################################
best_a = 9.8
best_b = -500
pred = best_a * data['Height'] + best_b

lm = LinearRegression()
lm.fit(data[['Height']], data['Weight'])

sns.scatterplot(x=data['Height'], y=data['Weight'], color='gray')
sns.lineplot(x=data['Height'], y=pred, color='red', label='Manual Fit')
sns.lineplot(x=data['Height'], y=lm.predict(data[['Height']]), color='green', label='Sklearn Fit')



# #############################################################
# Machine learning for linear regression
# #############################################################

X = torch.tensor(data['Height'].values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(data['Weight'].values, dtype=torch.float32).reshape(-1, 1)

# Define parameters a and b directly
a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Define loss function (Mean Absolute Error)
criterion = nn.L1Loss()
optimizer = optim.Adam([a, b], lr=0.1)

# Training loop
num_epochs = 1000
losses = []

for epoch in tqdm(range(num_epochs), desc="Training Progress"):
    predictions = a * X + b
    loss = criterion(predictions, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    


# Final parameters
final_a = a.item()
final_b = b.item()
pred_torch = (a * X + b).detach().numpy().flatten()

sns.scatterplot(x=data['Height'], y=data['Weight'], color='gray')
sns.lineplot(x=data['Height'], y=pred, color='red', label='Manual Fit')
sns.lineplot(x=data['Height'], y=lm.predict(data[['Height']]), color='green', label='Sklearn Fit')
sns.lineplot(x=data['Height'], y=pred_torch, color='blue', label='PyTorch Fit')


