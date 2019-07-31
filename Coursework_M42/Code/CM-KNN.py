# K-Nearest Neighbours

# Importing libraries
import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

path='../Data/Classification Models/'

# Load data
X_train = pd.read_csv(path+'X_train.csv')
X_test = pd.read_csv(path+'X_test.csv')
y_train = pd.read_csv(path+'y_train.csv')

# There's too many features (112),
# This selects the best K of them
k_best_model = SelectKBest(f_classif, k=50).fit(X_train, y_train['EpiOrStroma'])

# Transform data so it only has the best
# k features
X_new = k_best_model.transform(X_train)


# Test best params using grid search
params = {'n_neighbors' :[1, 5, 10, 20, 30, 40],
          'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=5, n_jobs=3)
grid_search.fit(X_new, y_train['EpiOrStroma'])
print("Best parameters for KNN:")
print(grid_search.best_params_)
print("KNN scores: ")
print(pd.DataFrame(grid_search.cv_results_)[['params', 'mean_test_score']])

# Train model using parameters found using grid search
model = KNeighborsClassifier(n_neighbors = 20, weights = 'uniform')\
    .fit(X_new, y_train['EpiOrStroma'])

# Use cross validation to evaluate model
cv = cross_val_score(model, X_new, y_train['EpiOrStroma'], cv=5)
print("10-fold cross validation accuracy: ", cv)
print("Mean cross validation accuracy: ", np.mean(cv))

# Transform test to have same number of 
# features as train, and predict y from it
y_pred=model.predict(k_best_model.transform(X_test))


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('KNN.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.