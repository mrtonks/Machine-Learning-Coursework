# Random Forest Regression Model

# Importing the libraries
import numpy as np # Maths
import matplotlib.pyplot as plt # Plot charts
import pandas as pd # Import/manage datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Importing the datasets
# Train datasets
X = np.array(pd.read_csv('../../Data/Regression Models/X_train.csv'))
y = np.array(pd.read_csv('../../Data/Regression Models/y_train.csv'))

# Remove index column from y array
y = y[:, 1]
# Split original train data into train and test data
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                     test_size = 0.25, 
                                                     random_state = 2018)
# Make test datasets, the validation dataset
X_test = np.array(pd.read_csv('../../Data/Regression Models/X_test.csv'))


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(random_state=42)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=2018, n_jobs=-1,
                              return_train_score=True)
# Fit the random search model
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_val, y_val)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_val, y_val)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [False],
    'max_depth': [70, 80, 90],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [700, 900, 1500]
}

# Create a based model
rf = RandomForestRegressor(random_state=2018)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, 
                          return_train_score=True)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_val, y_val)


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

# Another round of grid
param_grid = {
    'bootstrap': [False],
    'max_depth': [26, 27, 29],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1],
    'min_samples_split': [3, 4],
    'n_estimators': [289, 290, 291]
}

#{'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 285}

# Create a base model
rf = RandomForestRegressor(random_state=2018)

# Instantiate the grid search model
grid_search_final = GridSearchCV(estimator = rf, param_grid = param_grid, 
                                 cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)

grid_search_final.fit(X_train, y_train)

print(grid_search_final.best_params_)

best_grid_final = grid_search_final.best_estimator_
grid_final_accuracy = evaluate(best_grid_final, X_val, y_val)


#best_hyperparams = {
#    'bootstrap': False, 
#    'max_depth': 26, 
#    'max_features': 'sqrt', 
#    'min_samples_leaf': 1, 
#    'min_samples_split': 4, 
#    'n_estimators': 289
#}

regressor = RandomForestRegressor(bootstrap=False, max_depth=26, max_features='sqrt',
                                  min_samples_leaf=1, min_samples_split=4, n_estimators=289,
                                  random_state=2018)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


grid_final_accuracy = evaluate(best_grid_final, X_val, y_val)



test_header = "Id,PRP"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('random_forest_regression_with_289_trees.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")