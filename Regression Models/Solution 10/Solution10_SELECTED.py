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

""" Save the CSV file with predictions """
def saveCSV(file_name, test, pred):
    test_header = "Id,PRP"
    n_points = test.shape[0]
    y_pred_pp = np.ones((n_points, 2))
    y_pred_pp[:, 0] = range(n_points)
    y_pred_pp[:, 1] = pred
    np.savetxt(file_name, y_pred_pp, fmt='%d', delimiter=",",
               header=test_header, comments="")


# Importing the datasets
# Train datasets
X_train = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y_train = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

# Remove index column from y array
y_train = y_train[:, 1]

# Test datasets
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))

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
#0         3.6 Constant
#1         1.2 MYCT
#2         3.0 MMIN
#3         3.3 MMAX
#4         1.8 CACH
#5         2.0 CHMIN - Remove because P value is above 0.5
#6         1.8 CHMAX - Remove this column for multicollinearity
# For the same purpose

# Getting memory averages from the mmax and mmin for train and test sets
# This is trial and error on trying to improve the predictions
# and after observing other polynomials predictions
m_avg_train = (X_train[:, 1] + X_train[:, 2]) / 2 
m_avg_test = (X_test[:, 1] + X_test[:, 2]) / 2

# Drop columns MMIN, MMAX, CHMIN and CHMAX
X_train_opt = X_train[:, [0, 3]]
X_test_opt = X_test[:, [0, 3]]
# Get average of main memories min and max for improvement of predictions
X_train_opt = np.column_stack((X_train_opt, m_avg_train))
X_test_opt = np.column_stack((X_test_opt, m_avg_test))

# Create a new pipeline with Polynomial Features as preprocessor
# and Linear Regression as model
model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression())
# Fit the pipeline with the polynomial features
model.fit(X_train_opt, y_train)

# Make the predictions
y_pred = model.predict(X_test_opt)

saveCSV('polynomial_regression_grade2_remove_chmin_chmax_avgm.csv', X_test, y_pred)