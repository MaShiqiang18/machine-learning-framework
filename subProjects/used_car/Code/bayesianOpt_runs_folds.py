# coding=gbk

import pandas as pd
from bayes_opt import BayesianOptimization
from trainModels.main import run_train
from subProjects.used_car.Code.config import ParamConfig
import os


config = ParamConfig(mark='demo')
XY_train = pd.read_csv(config.path_data.path_train_XY, index_col=0).iloc[:200, :]
X_test = pd.read_csv(config.path_data.path_test_XY, index_col=0).iloc[:50, :]


X_train = XY_train.iloc[:, :-1]
y_train = XY_train.iloc[:, -1]
X_test = X_test.copy()

X_train_all = XY_train.iloc[:, :-1]
Y_train_all = XY_train.iloc[:, -1]
target_col = config.data_label


def compute_score(num_leaves, max_depth, min_child_samples):
    lgb_param = {'num_leaves': int(num_leaves),
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': int(max_depth),
             'learning_rate': 0.01,
             "min_child_samples": int(min_child_samples),
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'mae',
             "lambda_l1": 0.1,
             "verbosity": -1}


    pre_sumRuns_train, pre_sumRuns_test, score_final = run_train.train_model(XY_train, X_test, lgb_param, config)
    print(score_final)
    return score_final


rf_bo = BayesianOptimization(
    compute_score,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'min_child_samples': (2, 100)
    }
)

rf_bo.maximize()
haha = rf_bo.res['max']
print(haha)

