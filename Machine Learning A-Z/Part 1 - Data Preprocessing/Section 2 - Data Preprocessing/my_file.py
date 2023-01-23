#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset
dataset = pd.read_csv("Data.csv")

#Assign inputs and outputs to x and y
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#Take care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Encode categorical data: country and purchased (turn words, like countries, into numbers)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting into Training and Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

# Feature scaling: put all values in same range (-1 to 1)
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
