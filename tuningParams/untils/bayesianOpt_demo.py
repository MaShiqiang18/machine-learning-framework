# coding=gbk

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from lightgbm.sklearn import LGBMRegressor
from bayes_opt import BayesianOptimization
import os


path_up = os.path.abspath('../../')
data_path = path_up + r'/subProjects/used_car/Data'
X_train_path = data_path + '/used_car_80features_train.csv'
X_test_path = data_path + r'/used_car_80features_test.csv'

XY_train = pd.read_csv(X_train_path, index_col=0)
X_test = pd.read_csv(X_test_path, index_col=0)

X_train = XY_train.iloc[:,:-1]
y_train = XY_train.iloc[:,-1]
X_test = X_test.copy()


def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=X_train, y=y_train, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()

    return 1 - val


rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)


rf_bo.maximize()
haha = rf_bo.res['max']
print(haha)
