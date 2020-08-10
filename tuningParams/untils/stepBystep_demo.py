# coding=gbk

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from lightgbm.sklearn import LGBMRegressor
import os


def compute_score(model, X_train, y_train, metrics):
    score = np.mean(cross_val_score(model, X=X_train, y=y_train, verbose=0, cv=5, scoring=make_scorer(metrics)))
    return score


path_up = os.path.abspath('../../')
data_path = path_up + r'/subProjects/used_car/Data'
X_train_path = data_path + '/used_car_80features_train.csv'
X_test_path = data_path + r'/used_car_80features_test.csv'

XY_train = pd.read_csv(X_train_path, index_col=0)
X_test = pd.read_csv(X_test_path, index_col=0)

X_train = XY_train.iloc[:,:-1]
y_train = XY_train.iloc[:,-1]
X_test = X_test.copy()


best_obj = dict()
objective = ['regression']
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(
        cross_val_score(model, X=X_train, y=y_train, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score

best_leaves = dict()
num_leaves = [3, 5, 7]
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0], num_leaves=leaves)
    score = np.mean(
        cross_val_score(model, X=X_train, y=y_train, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score

best_depth = dict()
max_depth = [5, 7, 9]
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x: x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x: x[1])[0],
                          max_depth=depth)
    score = np.mean(
        cross_val_score(model, X=X_train, y=y_train, verbose=0, cv=5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score

best_params = {
    'objective': min(best_obj.items(), key=lambda x: x[1])[0],
    'num_leaves': min(best_leaves.items(), key=lambda x: x[1])[0],
    'max_depth': min(best_depth.items(), key=lambda x: x[1])[0]
}
print(best_params)