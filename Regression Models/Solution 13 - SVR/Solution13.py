# Support Vector Regression

# Importing the libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Plot charts
import pandas as pd # Import/manage datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the datasets
# Train datasets
X_train = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y_train = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

y_train = y_train[:, 1]
y_train = y_train.reshape(168, 1)

# Test datasets
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))

#m_avg_train = (X_train[:, 1] + X_train[:, 2]) / 2 
#m_avg_test = (X_test[:, 1] + X_test[:, 2]) / 2 
#
#X_train = X_train[:, [0,3]]
#X_train = np.column_stack((X_train, m_avg_train))
#
#X_test = X_test[:, [0,3]]
#X_test = np.column_stack((X_test, m_avg_test))

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)

X_test = sc_X.transform(X_test)

regressor = SVR(kernel='linear')
regressor.fit(X_train, y_train)

y_pred = sc_y.inverse_transform(regressor.predict(X_test))

#test_header = "Id,PRP"
#n_points = X_test.shape[0]
#y_pred_pp = np.ones((n_points, 2))
#y_pred_pp[:, 0] = range(n_points)
#y_pred_pp[:, 1] = y_pred
#np.savetxt('SVR_with_rbf.csv', y_pred_pp, fmt='%d', delimiter=",",
#           header=test_header, comments="")

