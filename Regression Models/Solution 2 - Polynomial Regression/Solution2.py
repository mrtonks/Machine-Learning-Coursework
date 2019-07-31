# Polynomial Regression

# Importing the libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Plot charts
import pandas as pd # Import/manage datasets

# Importing the datasets
# Train datasets
X_train = np.array(pd.read_csv('../../Data/X_train.csv'))
y_train = np.array(pd.read_csv('../../Data/y_train.csv'))

#X_train = X_train[:, :4]
y_train = y_train[:, 1]

# Test dataset
X_test = np.array(pd.read_csv('../../Data/X_test.csv'))

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)

y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
   
test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('my_submission.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

