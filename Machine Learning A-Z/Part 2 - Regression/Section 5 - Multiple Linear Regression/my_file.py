# %% Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %% Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# %% Encoding Countries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct  = ColumnTransformer([("encoder", OneHotEncoder(), [3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

# %% Avoiding Dummy Variable Trap: some libraries require first column does not exist yeet
x = x[:, 1:]

# %% Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



# %% Feature Scaling: not needed for Multiple Linear Regression b/c library takes care of that
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""


# %% Fit Multiple Linear Regression Model to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# %% Predicting Test Set Results
y_pred = regressor.predict(x_test)

# %% Build Optimal Model Using Backward Elimination
import statsmodels.api as sm
x = np.append(arr = np.ones((50,1)).astype(int), values=x, axis = 1)
x_opt = x[:, [0,1,2,3,4,5]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()         # look for P>|t|

# %% trash index 2, since pval is high
x_opt = x[:, [0,1,3,4,5]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# %% trash x1 index 1
x_opt = x[:, [0,3,4,5]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# %% trash x2 index 4
x_opt = x[:, [0,3,5]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

# %% trash x2 index 5
x_opt = x[:, [0,3]]
x_opt = np.array(x_opt, dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()
