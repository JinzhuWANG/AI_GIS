import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression



# Get data
data = pd.read_csv('data/circle_data.csv')
sns.scatterplot(data=data, x='x1', y='y1', color='blue', label='Set 1')
sns.scatterplot(data=data, x='x2', y='y2', color='red', label='Set 2')




# Convert to radius representation
data_r = data.copy() - 10
sns.scatterplot(data=data_r, x='x1', y='y1', color='blue', label='Set 1')
sns.scatterplot(data=data_r, x='x2', y='y2', color='red', label='Set 2')


data_r['radius_1'] = np.sqrt(data_r['x1']**2 + data_r['y1']**2)
data_r['radius_2'] = np.sqrt(data_r['x2']**2 + data_r['y2']**2)
data_r['angle_1'] = np.arctan2(data_r['y1'], data_r['x1'])
data_r['angle_2'] = np.arctan2(data_r['y2'], data_r['x2'])

sns.scatterplot(data=data_r, x='radius_1', y='angle_1', color='blue', label='Set 1')
sns.scatterplot(data=data_r, x='radius_2', y='angle_2', color='red', label='Set 2')


# Plot the decision line in the radius coordinates
lm = LinearRegression()
lm.fit([[5.7], [6.3]], [[-3], [3]])
lm.coef_, lm.intercept_

pred_radius = np.arange(5.7, 6.3, 0.01)
pred_angle = lm.predict(pred_radius.reshape(-1, 1)).flatten()

sns.scatterplot(data=data_r, x='radius_1', y='angle_1', color='blue', label='Set 1')
sns.scatterplot(data=data_r, x='radius_2', y='angle_2', color='red', label='Set 2')
sns.lineplot(x=pred_radius, y=pred_angle, color='green', label='Decision Line')




# Visualize decision line in the original coordinates
pred_x = pred_radius * np.cos(pred_angle)
pred_y = pred_radius * np.sin(pred_angle)
pred_x_original = pred_x + 10
pred_y_original = pred_y + 10


sns.scatterplot(data=data, x='x1', y='y1', color='blue', label='Set 1')
sns.scatterplot(data=data, x='x2', y='y2', color='red', label='Set 2')
sns.lineplot(x=pred_x_original, y=pred_y_original, color='green', sort=False, label='Decision Line')


pred_df = pd.DataFrame({
    'x': pred_x_original,
    'y': pred_y_original
})

pred_df.to_csv('data/circle_data_decision_line.csv')
