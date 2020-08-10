# coding=gbk

from subProjects.used_car.Code.config import ParamConfig
from trainModels.main.run_train import train_model
import pandas as pd
import numpy as np
import os


print('\n\n=============================>  准备数据集\n')
# # print('subProjects/used_car/Data')
# # 获得当前工作目录
# path_ori = os.getcwd()
# # path_ori = os.path.abspath('.')
# # path_ori = os.path.abspath(os.curdir)
#
# # 获得当前工作目录的父目录
# path_up = os.path.abspath('..')
#
# ## 原始数据
# # train_path = path_up + r'/Data/used_car_train_20200313.csv'
# # test_path = path_up + r'/Data/used_car_testB_20200421.csv'
# # Train_data = pd.read_csv(train_path, sep=' ', encoding='gb18030').iloc[:10000, :]
# # TestA_data = pd.read_csv(test_path, sep=' ', encoding='gb18030').iloc[:10000, :]
# # # #Train_data['price'] = np.log1p(Train_data['price'])
#
# ## 特征工程后的数据
# X_train_path = path_up + r'/Data/used_car_80features_train.csv'
# X_test_path = path_up + r'/Data/used_car_80features_test.csv'
# Train_data = pd.read_csv(X_train_path, index_col=0)
# TestA_data = pd.read_csv(X_test_path, index_col=0)
#
# #合并数据集
# concat_data = pd.concat([Train_data, TestA_data])
# concat_data.reset_index(drop=True, inplace=True)
# print('训练集大小:', Train_data.shape)
# print('测试集大小:', TestA_data.shape)
# print('训练集测试集合并后大小:', concat_data.shape)


# param = {'num_leaves': 30,
#          'min_data_in_leaf': 30,
#          'objective': 'regression',
#          'max_depth': -1,
#          'learning_rate': 0.01,
#          "min_child_samples": 30,
#          "boosting": "gbdt",
#          "feature_fraction": 0.9,
#          "bagging_freq": 1,
#          "bagging_fraction": 0.9,
#          "bagging_seed": 11,
#          "metric": 'mae',
#          "lambda_l1": 0.1,
#          "verbosity": -1}

param = {
    'epochs': 20,
    'batch_size': 12000,
    'show_fig': False,
    'loss': 'mean_absolute_error',
    'optimizer': 'adam',
    'metrics': 'mae'
}

## initialize a param config
# config = ParamConfig(mark='demo', cleanMarkData=True)
config = ParamConfig(mark='demo')
XY_train = pd.read_csv(config.path_data.path_train_XY, index_col=0).iloc[:200, :]
X_test = pd.read_csv(config.path_data.path_test_XY, index_col=0).iloc[:50, :]
pre_sumRuns_train, pre_sumRuns_test, score_final = train_model(XY_train, X_test, param, config)
print(score_final)
