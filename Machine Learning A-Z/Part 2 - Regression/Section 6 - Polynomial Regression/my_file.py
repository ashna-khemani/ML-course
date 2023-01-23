# %% Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# %% Fit Linear Regression to x and y
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# %% Fit Polynomial Regression to x and y
    # make x_poly which has x, x^2, x^3, x^4
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

    # fit linear model to ind. var. with multiple features (x_poly) and y
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


# %% Viualize Linear Regression
plt.scatter(x,y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Truth Or Bluff (Linear)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# %% Visualize Polynimial Regression
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

# %% Predict Level 6.5 salary with linear model
lin_reg.predict([[6.5]])

# %% Predict Level 6.5 Salary with polynomial model
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

# %%
