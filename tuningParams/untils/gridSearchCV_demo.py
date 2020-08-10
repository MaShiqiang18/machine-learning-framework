# coding=gbk

import pandas as pd
import numpy as np
from lightgbm.sklearn import LGBMRegressor
from sklearn.model_selection import GridSearchCV
import os


def grid_Search_cv(train_X, train_y, model, parameters ):
    grid_search = GridSearchCV(model, parameters, cv=5)
    grid_search .fit(train_X, train_y)
    return grid_search.best_params_, grid_search.best_estimator_


path_up = os.path.abspath('../../')
data_path = path_up + r'/subProjects/used_car/Data'
X_train_path = data_path + '/used_car_80features_train.csv'
X_test_path = data_path + r'/used_car_80features_test.csv'

XY_train = pd.read_csv(X_train_path, index_col=0)
X_test = pd.read_csv(X_test_path, index_col=0)

X_train = XY_train.iloc[:,:-1]
y_train = XY_train.iloc[:,-1]
X_test = X_test.copy()


objective = ['regression']
num_leaves = [3, 5, 7]
max_depth = [5, 7, 9]
parameters = {'objective': objective, 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
best_params, best_estimator = grid_Search_cv(X_train, y_train, model, parameters )