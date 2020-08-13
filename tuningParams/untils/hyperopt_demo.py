# coding=gbk

import hyperopt
import pandas as pd
import os
from hyperopt import fmin, tpe, hp, partial
import lightgbm as lgb

path_up = os.path.abspath('../../')
data_path = path_up + r'/subProjects/used_car/Data'
X_train_path = data_path + '/used_car_80features_train.csv'
X_test_path = data_path + r'/used_car_80features_test.csv'

XY_train = pd.read_csv(X_train_path, index_col=0)
X_test = pd.read_csv(X_test_path, index_col=0)

X_train = XY_train.iloc[:,:-1]
y_train = XY_train.iloc[:,-1]
X_test = X_test.copy()

train_all_data = lgb.Dataset(data=X_train, label=y_train)

def hyperopt_objective(params):

    model = lgb.LGBMRegressor(
        num_leaves=31,
        max_depth=int(params['max_depth']) + 5,
        learning_rate=params['learning_rate'],
        objective='regression',
        eval_metric='rmse',
        nthread=-1,
    )

    num_round = 10
    res = lgb.cv(model.get_params(),train_all_data, num_round, nfold=5, metrics='rmse',early_stopping_rounds=10)

    return min(res['rmse-mean']) # as hyperopt minimises
# 这里的warnings实在太多了，我们加入代码不再让其显示
import warnings
warnings.filterwarnings("ignore")

from numpy.random import RandomState

params_space = {
    'max_depth': hyperopt.hp.randint('max_depth', 6),
    'learning_rate': hyperopt.hp.uniform('learning_rate', 1e-3, 5e-1),
}

trials = hyperopt.Trials()

best = hyperopt.fmin(
    hyperopt_objective,
    space=params_space,
    algo=hyperopt.tpe.suggest,
    max_evals=50,
    trials=trials,
    rstate=RandomState(123)
)

print("\n展示hyperopt获取的最佳结果，但是要注意的是我们对hyperopt最初的取值范围做过一次转换")
print(best)