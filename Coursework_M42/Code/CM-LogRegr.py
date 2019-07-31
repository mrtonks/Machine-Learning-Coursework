# Logaristic Regression

# Importing libraries
import numpy as np
import pandas as pd

# Sklearn libraries
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, GridSearchCV

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


# Fit model
model = LogisticRegression(class_weight = 'balanced',
                          random_state=1)\
    .fit(X_new, y_train['EpiOrStroma'])


# Use cross validation to evaluate model
cv = cross_val_score(model, X_new, y_train['EpiOrStroma'], cv=5)
print("5-fold cross validation accuracy: ", cv)
print("Mean cross validation accuracy: ", np.mean(cv))

# Transform test to have same number of 
# features as train, and predict y from it
y_pred = model.predict(k_best_model.transform(X_test))


# Arrange answer in two columns. First column (with header "Id") is an
# enumeration from 0 to n-1, where n is the number of test points. Second
# column (with header "EpiOrStroma" is the predictions.
test_header = "Id,EpiOrStroma"
n_points = X_test.shape[0]
y_pred_pp = np.ones((n_points, 2))
y_pred_pp[:, 0] = range(n_points)
y_pred_pp[:, 1] = y_pred
np.savetxt('LogisticRegression.csv', y_pred_pp, fmt='%d', delimiter=",",
           header=test_header, comments="")

# Note: fmt='%d' denotes that all values should be formatted as integers which
# is appropriate for classification. For regression, where the second column
# should be floating point, use fmt='%d,%f'.

