# %% Import libraries
import pandas as pd
import numpy as np

# %% Dataset
dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, -1].values

# Split into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# %% Fit logistic regression model on training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# %% Make predictions on test set
y_pred = classifier.predict(X_test)
print("Predictions on test set: \n", np.c_[y_pred, y_test])

# %% Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %% Cross Validation
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(estimator=classifier, X=X, y=Y)
print(cv_score)
