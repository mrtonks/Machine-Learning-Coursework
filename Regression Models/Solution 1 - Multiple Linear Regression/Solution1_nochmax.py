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

#plt.scatter(X_train, y_train)

# Test datasets
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))
#X_test = X_test[:, :4]

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred = np.array(y_pred, dtype=np.int64)

# Building the optimal model using Backward Elimination
# SL = 0.05 = 5%
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

r_squared = regressor.score(X_train, y_train)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
vif.round(1)

X_train_2 =  X_train[:, 0:5]
X_test_2 = X_test[:, 0:5]

regressor.fit(X_train_2, y_train)

y_pred = regressor.predict(X_test_2)
y_pred = np.array(y_pred, dtype=np.int64)

#X = np.append(arr = np.ones((X_train.shape[0], 1)).astype(int), values = X_train, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_opt, i) for i in range(X_opt.shape[1])]
vif.round(1)

X_train_2 =  X_train[:, 0:4]
X_test_2 = X_test[:, 0:4]

regressor.fit(X_train_2, y_train)

y_pred = regressor.predict(X_test_2)
y_pred = np.array(y_pred, dtype=np.int64)

X_opt = X[:, [0, 1, 2, 3, 4]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_opt, i) for i in range(X_opt.shape[1])]
vif.round(1)

m_avg_train = (X_opt[:, 2] + X_opt[:, 3]) / 2 
m_avg_test = (X_test_2[:, 1] + X_test_2[:, 2]) / 2

X_opt = X[:, [0, 1, 4]]
X_opt = np.column_stack((X_opt, m_avg_train))
X_test_2 = np.column_stack((X_test_2, m_avg_test))

regressor.fit(X_opt[:, 1:], y_train)

y_pred = regressor.predict(X_test_2[:, [0, 3, 4]])
y_pred = np.array(y_pred, dtype=np.int64)


regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_opt, i) for i in range(X_opt.shape[1])]
vif.round(1)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
#regressor = LinearRegression()
#regressor.fit(X_train, y_train)

model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
model.fit(X_opt[:, 1:], y_train)

y_pred = model.predict(X_test_2[:, [0, 3, 4]])

#X_opt = X[:, [0, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#
#X_opt = X[:, [0, 3, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#
#X_opt = X[:, [0, 3]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#
#test_header = "Id,PRP"
#n_points = X_test.shape[0]
#y_pred_pp = np.ones((n_points, 2))
#y_pred_pp[:, 0] = range(n_points)
#y_pred_pp[:, 1] = y_pred
#np.savetxt('polynomial_regression_grade4_remove_chmin_chmax_avgm.csv', y_pred_pp, fmt='%d', delimiter=",",
#           header=test_header, comments="")