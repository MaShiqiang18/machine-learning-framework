# coding=gbk

from subProjects.used_car.Code.config import ParamConfig
from predictByModels.run_predict import predict_by_models
import pandas as pd


## initialize a param config
config = ParamConfig(mark='demo')
XY_train = pd.read_csv(config.path_data.path_train_XY, index_col=0).iloc[:200, :]
X_test = pd.read_csv(config.path_data.path_test_XY, index_col=0).iloc[:50, :]
pre_sumRuns_train, pre_sumRuns_test, score_final = predict_by_models(XY_train, X_test, config)
print(score_final)
