# Random Forest Regression Model

# Importing the libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Plot charts
import pandas as pd # Import/manage datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

feature_list = ['MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX']

# Importing the datasets
# Train datasets
X = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

#X_train = X_train[:, :4]
y = y[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                     test_size = 0.25, 
                                                     random_state = 42)
# Test datasets
X_validate = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))


# Create a model
rf = RandomForestRegressor(bootstrap=False, max_depth=26, max_features='sqrt',
                                  min_samples_leaf=1, min_samples_split=3, n_estimators=289,
                                  random_state=42)
# Train the model
rf.fit(X_train, y_train)

# Use the forest's predict method on the test data
predictions = rf.predict(X_test)

# Calculate the absolute errors
errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]


# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(bootstrap=False, max_depth=26, max_features='sqrt',
                                  min_samples_leaf=1, min_samples_split=3, n_estimators=289,
                                  random_state=42)
# Extract the two most important features
important_indices = [feature_list.index('MMAX'), feature_list.index('MMIN'),
                     feature_list.index('CACH'), feature_list.index('MYCT')]
train_important = X_train[:, important_indices]
test_important = X_test[:, important_indices]
validation_important = X_validate[:, important_indices]
# Train the random forest
rf_most_important.fit(train_important, y_train)
# Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - y_test)
# Display the performance metrics
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = np.mean(100 * (errors / y_test))
accuracy = 100 - mape
print('Accuracy:', round(accuracy, 2), '%.')


y_pred = rf_most_important.predict(validation_important)

test_header = "Id,PRP"
n_points = X_validate.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('random_forest_regression_with_289_trees_removecols.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")