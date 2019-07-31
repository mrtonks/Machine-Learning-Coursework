# Random Forest Regression Model

# Importing the libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Plot charts
import pandas as pd # Import/manage datasets
from sklearn.ensemble import RandomForestRegressor

# Importing the datasets
# Train datasets
X_train = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y_train = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

#X_train = X_train[:, :4]
y_train = y_train[:, 1]

#plt.scatter(X_train, y_train)

# Test datasets
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))

regressor = RandomForestRegressor(n_estimators=290, random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#test_header = "Id,PRP"
#n_points = X_test.shape[0]
#y_pred_pp = np.ones((n_points, 2))
#y_pred_pp[:, 0] = range(n_points)
#y_pred_pp[:, 1] = y_pred
#np.savetxt('random_forest_regression_with_290_trees.csv', y_pred_pp, fmt='%d', delimiter=",",
#           header=test_header, comments="")