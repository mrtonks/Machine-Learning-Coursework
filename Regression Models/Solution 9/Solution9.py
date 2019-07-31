# Multiple Linear Regression

# Importing the libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Plot charts
import pandas as pd # Import/manage datasets

# Importing the datasets
# Train datasets
X_train = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y_train = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

#X_train = X_train[:, :4]
y_train = y_train[:, 1]

# Test datasets
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))
#X_test = X_test[:, :4]

X_train_size = len(X_train)
X_test_size = len(X_test)

X = np.append(X_train, X_test, axis=0)

# Rescale X
X = (X - np.mean(X)) / np.std(X)

X_train = X[:X_train_size, :]
X_test = X[:X_test_size, :]

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred, dtype=np.int64)

# Building the optimal model using Backward Elimination
# SL = 0.05 = 5%
#import statsmodels.formula.api as sm
#X = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
#regressor_OLS.summary()



#test_header = "Id,PRP"
#n_points = X_test.shape[0]
#y_pred_pp = np.ones((n_points, 2))
#y_pred_pp[:, 0] = range(n_points)
#y_pred_pp[:, 1] = y_pred
#np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
#           header=test_header, comments="")