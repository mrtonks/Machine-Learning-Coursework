# Multiple Linear Regression

# Importing the libraries
import numpy as np # Maths
import pandas as pd # Import/manage datasets

# Sklearn libraries
from sklearn.linear_model import LinearRegression # Linear Regression
from sklearn.preprocessing import PolynomialFeatures # Polinomial features for preprocessing
from sklearn.pipeline import make_pipeline # Builds a pipeline

# Stats Models
import statsmodels.formula.api as sm # For OLS regressor
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Importing the datasets
# Train datasets
X_train = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y_train = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

# Remove index column from y array
y_train = y_train[:, 1]

# Test datasets
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred, dtype=np.int64)

# Observe the Ordinary Least Squares Regression for coeficients and results
X = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
# This result mentions strong multicollinearity

# Observe the Variance Inflation Factor for multicollinearity
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_opt, i) for i in range(X_opt.shape[1])]
vif.round(1)

#   VIF Factor
#0         3.6
#1         1.2
#2         3.0
#3         3.3
#4         1.8
#5         2.0 I'll remove this column which is related - CHMIN
#6         1.8 Remove this column - CHMAX
# For the same purpose

# Getting memory averages from the mmax and mmin for train and test sets
# This is trial and error on trying to improve the predictions
# and after observing other polynomials predictions
m_avg_train = (X_opt[:, 2] + X_opt[:, 3]) / 2 
m_avg_test = (X_test[:, 1] + X_test[:, 2]) / 2

X_opt = X[:, [0, 1, 4]]
X_test_opt = X_test[:, [0, 1, 4]]
X_opt = np.column_stack((X_opt, m_avg_train))
X_test_opt = np.column_stack((X_test_opt, m_avg_test))

# Fit the pipeline with the polynomial features
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model.fit(X_opt[:, 1:], y_train)

y_pred = model.predict(X_test_opt[:, 1:])

#test_header = "Id,PRP"
#n_points = X_test.shape[0]
#y_pred_pp = np.ones((n_points, 2))
#y_pred_pp[:, 0] = range(n_points)
#y_pred_pp[:, 1] = y_pred
#np.savetxt('polynomial_regression_grade2_remove_chmin_chmax_avgm.csv', y_pred_pp, fmt='%d', delimiter=",",
#           header=test_header, comments="")