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

m_avg_train = (X_train[:, 1] + X_train[:, 2]) / 2 
ch_avg_train = (X_train[:, 4] + X_train[:, 5]) / 2 
X_train_fix = np.delete(X_train, 1, 1)
X_train_fix = np.delete(X_train_fix, 1, 1)
X_train_fix = np.delete(X_train_fix, 2, 1)
X_train_fix = np.delete(X_train_fix, 2, 1)

X_train_fix = np.column_stack((X_train_fix, m_avg_train))
X_train_fix = np.column_stack((X_train_fix, ch_avg_train))

m_avg_test = (X_test[:, 1] + X_test[:, 2]) / 2 
ch_avg_test = (X_test[:, 4] + X_test[:, 5]) / 2 
X_test_fix = np.delete(X_test, 1, 1)
X_test_fix = np.delete(X_test_fix, 1, 1)
X_test_fix = np.delete(X_test_fix, 2, 1)
X_test_fix = np.delete(X_test_fix, 2, 1)

X_test_fix = np.column_stack((X_test_fix, m_avg_test))
X_test_fix = np.column_stack((X_test_fix, ch_avg_test))

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_fix, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test_fix)
y_pred = np.array(y_pred, dtype=np.int64)


test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('multiple_linear_regression_mavg_chavg.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")