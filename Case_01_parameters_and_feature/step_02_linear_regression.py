import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Read the dataset
data = pd.read_csv('data/height-weight-20.csv')
sns.scatterplot(x=data['Height'], y=data['Weight'])



# Gusses the a,b for "weight = a * height + b"
a = 10
b = -500

pred = a * data['Height'] + b
diff = pred - data['Weight']
loss = abs(diff).mean()



# Adjust a
adjust_range_1 = np.arange(-0.25, 0, 0.02)

losses = []
for adjust in adjust_range_1:
    pred = (a + adjust) * data['Height'] + b
    diff = pred - data['Weight']
    loss = abs(diff).mean()
    losses.append(loss)
    
sns.scatterplot(x=adjust_range_1, y=losses)



# Adjust b
best_a = a - 0.2
adjust_range_b = np.arange(-1, 1, 0.1)

losses_b = []
for adjust in adjust_range_b:
    pred = best_a * data['Height'] + (b + adjust)
    diff = pred - data['Weight']
    loss = abs(diff).mean()
    losses_b.append(loss)

sns.scatterplot(x=adjust_range_b, y=losses_b)


# Best guess
best_b = b + 0
pred = best_a * data['Height'] + best_b
loss = abs(pred - data['Weight']).mean()

sns.scatterplot(x=data['Height'], y=data['Weight'])
sns.lineplot(x=data['Height'], y=pred, color='red')



# fit with sklearn
lm = LinearRegression()
lm.fit(data[['Height']], data['Weight'])
lm.coef_, lm.intercept_
loss = abs(lm.predict(data[['Height']]) - data['Weight']).mean()

sns.scatterplot(x=data['Height'], y=data['Weight'], color='gray')
sns.lineplot(x=data['Height'], y=pred, color='red', label='Best Guess')
sns.lineplot(x=data['Height'], y=lm.predict(data[['Height']]), color='green', label='Sklearn Fit')


