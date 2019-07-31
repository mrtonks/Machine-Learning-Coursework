# Random Forest Regression

# Importing the libraries
import numpy as np # Maths
import pandas as pd # Import/manage datasets

# Sklearn libraries
from sklearn.ensemble import RandomForestRegressor # Random Forest Regressor
# Randomized Search CV  - Randomized search on hyper parameteres
# GridSearhCV - Exhaustive search over a specified parameter values for an estimator
# train_test_split -  Split arrays or matrices into random train and test subsets
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split


# Function definitions
""" Get the random grid combinations for testing 
    with RandomizedSearchVC """
def getRandomGridCombinations():
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
    return random_grid

""" Evaluate the model against predictions  """
def evaluate(model, features, labels):
    predictions = model.predict(features)
    errors = abs(predictions - labels)
    mape = 100 * np.mean(errors / labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

""" Get best hyperparams from GridSearch """
def getBestHyperparams(params):
    param_grid = params
    
    # Create a based model
    rf = RandomForestRegressor(random_state=42)
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                              cv = 3, n_jobs = -1, verbose = 2, 
                              return_train_score=True)
    
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    
    # Print the best params from the randomized
    print(grid_search.best_params_)
    
    # Gets the best grid of params for the estimator of Random Forest
    # Has the highest score
    best_grid = grid_search.best_estimator_
    grid_accuracy = evaluate(best_grid, X_val, y_val)
    
    
    print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
    return best_grid

    
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
path='../../Data/Regression Models/'

# Train datasets
X = np.array(pd.read_csv(path+'X_train.csv'))
y = np.array(pd.read_csv(path+'y_train.csv'))

# Remove index column from y array
y = y[:, 1]

# Split original train datasets into train and test data
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                     test_size = 0.25, 
                                                     random_state = 42)
# Importing test dataset
X_test = np.array(pd.read_csv(path+'X_test.csv'))


# Start trial and error for best combinations of Random Grid
# Reference 
# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
random_grid = getRandomGridCombinations()

# Use the Random Grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor(random_state=42)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 100, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

# Fit the random search model
rf_random.fit(X_train, y_train)

# Print best params to keep experimenting
print(rf_random.best_params_)

# Create a model to start evaluating and improving
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_val, y_val)

# Get the best random hyperparams from the randomized search 
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_val, y_val)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

# After seeing the best params from the random search, we start tweaking
# the params to get the best ones
    
# Repeat this search tweaking the hyperparams until satisfied
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [False],
    'max_depth': [60, 65, 70],
    'max_features': ['sqrt'],
    'min_samples_leaf': [1],
    'min_samples_split': [3, 5],
    'n_estimators': [1600, 1650, 1700]
}

# Get the best hyperparams grids
best_grid = getBestHyperparams(param_grid)

# Here we review the last hyperparams to check if these are better
# Or just take last ones

#best_hyperparams = {
#    'bootstrap': False, 
#    'max_depth': 26, 
#    'max_features': 'sqrt', 
#    'min_samples_leaf': 1, 
#    'min_samples_split': 3, 
#    'n_estimators': 289
#}

# Create a model with the best hyperparams found in previous steps
regressor = RandomForestRegressor(bootstrap=False, max_depth=26, max_features='sqrt',
                                  min_samples_leaf=1, min_samples_split=3, n_estimators=289,
                                  random_state=42)
# Fit the model with the train datasets
regressor.fit(X_train, y_train)
# Predict the results with the test dataset
y_pred = regressor.predict(X_test)

# Evaluate the accuracy
grid_final_accuracy = evaluate(best_grid, X_test, y_pred)

saveCSV('random_forest_regression_with_1700_trees_seed_2018.csv', X_test, y_pred)

