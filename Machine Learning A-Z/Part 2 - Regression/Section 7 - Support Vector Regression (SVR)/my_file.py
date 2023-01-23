# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# %% Import Dataset
dataset = pd.read_csv("Position_Salaries.csv")
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values



# %% Reshape y so it's 2D array
y = y.reshape(len(y), 1)


# %% Feature Scaling x and y
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)

sc_y = StandardScaler()
y = sc_y.fit_transform(y)



# %% Train SVR Regressor on whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)



# %% Predict y value for x=6.5
sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])))


# %% Graph it
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x)), color = 'blue')
plt.title("Low Res")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()


# %% Higher resolution
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(sc_x.inverse_transform(x), sc_y.inverse_transform(y), color='red')
plt.plot(sc_x.inverse_transform(x_grid), sc_y.inverse_transform(regressor.predict(x_grid)), color='blue')
plt.title("High Res")
plt.xlabel("Level")
plt.ylabel("Salary")


# %%
