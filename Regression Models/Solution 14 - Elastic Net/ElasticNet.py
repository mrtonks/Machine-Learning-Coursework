# Importing the libraries
import numpy as np # Maths

# Plot charts
import seaborn as sns

import pandas as pd # Import/manage datasets

# data processing
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import ElasticNet # machine learning

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error # eval metric

# Importing the datasets
# Train datasets
X = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv')['PRP'])

# Test datasets
X_test= np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))

# Split the data
X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), 
                                                     test_size = 0.25, 
                                                     random_state = 2018)

# data preprocessing using sklearn Pipeline
pipeline = make_pipeline(PolynomialFeatures(degree=2, interaction_only=True), # multiply features together
                         StandardScaler()) # scale data

# fit and apply transform
X_train = pipeline.fit_transform(X_train)
# transform the validation set
X_val = pipeline.transform(X_val)
print('train shape:', X_train.shape, 'validation shape:', X_val.shape)


reg = ElasticNet(alpha=1.7)
reg.fit(X_train, y_train) # magic happens here
y_pred = reg.predict(X_val)
y_pred[y_pred < 0] = 0
print('Model MAE:', mean_absolute_error(y_val, y_pred))
print('Mean  MAE:', mean_absolute_error(y_val, np.full(y_val.shape, y.mean())))

# refit and predict submission data
X_train = pipeline.fit_transform(X)
X_test_last = pipeline.transform(X_test)
reg.fit(X_train, y)
y_pred = reg.predict(X_test_last)
y_pred[y_pred < 0] = 0

test_header = "Id,PRP"
n_points = X_test_last.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('ElasticNet_with_seed2018.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")