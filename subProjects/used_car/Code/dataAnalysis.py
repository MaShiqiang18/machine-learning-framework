# coding=gbk

import pandas as pd
import numpy as np
from subProjects.used_car.Code.config import ParamConfig
from analysis.main import run_analysis

config = ParamConfig(mark='demo')
print('\n=============================>  准备数据集\n')

train_path = config.path_data.dataOriPath + r'/used_car_train_20200313.csv'
test_path = config.path_data.dataOriPath + r'/used_car_testB_20200421.csv'
Train_data = pd.read_csv(train_path, sep=' ', encoding='gb18030').iloc[:10000, :]
TestA_data = pd.read_csv(test_path, sep=' ', encoding='gb18030').iloc[:10000, :]


data = Train_data
columns = Train_data.columns.tolist()

num_features = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4']
categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox']
labels = 'price'
run_analysis.analysis_main(data, columns, labels=labels, categorical_features=categorical_features, num_features=num_features)