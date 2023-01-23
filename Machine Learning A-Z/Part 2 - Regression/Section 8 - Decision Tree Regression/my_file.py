# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Get Dataset and split into x and y
dataset = pd.read_csv('Position_Salaries.csv')
x = np.array(dataset.iloc[:, 1:-1].values)
y = np.array(dataset.iloc[:, -1].values)

# %% Train Decision Tree Regression on Whole Dataset
from sklearn.tree import DecisionTreeRegressor as dtr
regressor = dtr(random_state = 0)
regressor.fit(x,y)

# %% Predict a value for 6.5
regressor.predict([[6.5]])

# %% Graph in high-res
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title("Position and Salary (DTR)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
