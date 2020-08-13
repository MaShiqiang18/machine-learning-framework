# coding=gbk

import pandas as pd
import numpy as np
from subProjects.used_car.Code.config import ParamConfig
from preProcess.untils import dataPreprocess
from newFeatures.untils import CodeFeatures, ReduceDimension
from newFeatures.main import produceFeatures as RFP
from preProcess.main import run_preprocess
from newFeatures.untils import NewFeatures, ScoreFeatures
from analysis.main import run_analysis
from analysis.untils import dataAnalysis

from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge

import os

mark = 'v1'
mark_text = ''
config = ParamConfig(mark=mark, mark_text=mark_text)
print('\n=============================>  ׼�����ݼ�\n')
# print('subProjects/used_car/Data')
# ��õ�ǰ����Ŀ¼
# path_ori = os.getcwd()
# path_ori = os.path.abspath('.')
# path_ori = os.path.abspath(os.curdir)
# ��õ�ǰ����Ŀ¼�ĸ�Ŀ¼
# path_up = os.path.abspath('../')

train_path = config.path_data.dataOriPath + r'/used_car_train_20200313.csv'
test_path = config.path_data.dataOriPath + r'/used_car_testB_20200421.csv'
Train_data = pd.read_csv(train_path, sep=' ', encoding='gb18030')
TestA_data = pd.read_csv(test_path, sep=' ', encoding='gb18030')
# #Train_data['price'] = np.log1p(Train_data['price'])


print('\n=============================>  ����Ԥ����\n')
# abnormal_index_train = pd.read_csv(config.path_data.dataOriPath + r'/abnormal_index.csv', index_col=0, header=0).index.tolist()
# # ɾ��ѵ�����е��쳣ֵ�������У�
# print('\n=============================>  ɾ���쳣ֵ\n')
# for i in abnormal_index_train:
#     Train_data.drop(i, axis=0, inplace=True)
# Train_data.reset_index(drop=True, inplace=True)
#�ϲ����ݼ�
concat_data = pd.concat([Train_data, TestA_data])
# ���������ȱʧֵ
# concat_data = concat_data.fillna(concat_data.mode().iloc[0,:])
concat_data.reset_index(drop=True, inplace=True)
print('ѵ������С:', Train_data.shape)
print('���Լ���С:', TestA_data.shape)
print('ѵ�������Լ��ϲ����С:', concat_data.shape)
cols = Train_data.columns.tolist()
label = 'price'
lastTrainIndex = Train_data.shape[0]-1

concat_data['notRepairedDamage'] = concat_data['notRepairedDamage'].replace('-', 0).astype('float16')
concat_data['power'] = concat_data['power'].map(lambda x: 600 if x > 600 else x)
concat_data['power'] = concat_data['power'].map(lambda x: 1 if x < 1 else x)
concat_data['v_13'] = concat_data['v_13'].map(lambda x: 6 if x > 6 else x)
concat_data['v_14'] = concat_data['v_14'].map(lambda x: 4 if x > 4 else x)


categorical_features = ['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox',
                        'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'creatDate', 'kilometer' ]

num_features = ['v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13',
                'v_14', 'power']

time_features = ['creatDate', 'regDate']

concat_data, abnormal_index_set = run_preprocess.preprocess_main(concat_data, categorical_features=categorical_features,
                                             num_features=num_features, time_features=time_features, fillnan=True)

# abnormal_index_train = [x for x in abnormal_index_set if x <= lastTrainIndex]
# abnormal_index_test = [x for x in abnormal_index_set if x > lastTrainIndex]


print('\n=============================>  �������̿�ʼ')
print('')
## �趨����
count_coding_features = ['regDate', 'creatDate', 'model', 'brand', 'regionCode', 'bodyType', 'fuelType', 'name',
                         'regDate_year', 'regDate_month', 'regDate_day', 'regDate_dayofweek', 'creatDate_month',
                         'creatDate_day', 'creatDate_dayofweek', 'kilometer']

comp_features = [['model', 'brand'],
                 ['model', 'regionCode'],
                 ['brand', 'regionCode']
                 ]

tocut_features = ['power']

onehot_features = []

cross_cat = ['model', 'brand','regDate_year']

cross_num = ['v_0', 'v_3', 'v_4', 'v_8', 'v_12', 'power']

## ���������
data_new = RFP.featureProcess_main(concat_data, cols, 'price', categorical_features=categorical_features,
                                   num_features=num_features, count_coding_features=count_coding_features,
                                   tocut_features=tocut_features, time_features=time_features,
                                   onehot_features=onehot_features, comp_features=comp_features,
                                   cross_cat=cross_cat, cross_num=cross_num)


# data_new = NewFeatures.count_coding(data_new, count_coding_features)

print('\n= = = = = = = = = = = = = = =>  �������̽���������\n')
# ��ɸѡһ��
cat_cols = ['SaleID', 'offerType', 'seller']
feature_cols = data_new.columns
numerical_cols = [col for col in feature_cols if col not in cat_cols]
numerical_cols = [col for col in numerical_cols if col not in ['price']]
## ��ǰ�����У���ǩ�й���ѵ�������Ͳ�������
X_data = data_new.iloc[:len(Train_data), :][numerical_cols]
Y_data = Train_data.loc[:, ['price']]
X_test = data_new.iloc[len(Train_data):, :][numerical_cols]



print('\n=============================>  ʹ��ƽ��ֵ����\n')
Feature_list = ['model', 'brand', 'name', 'regionCode']
X_train, X_test = CodeFeatures.meanEnocoding(X_data, Y_data, X_test, Feature_list)


print('\n=============================>  ʹ��Ŀ�����\n')
X_train['price'] = Y_data['price']
group_by_features = ['regionCode','brand','regDate_year','creatDate_year','kilometer','model']
col = 'price'
X_train, X_test = CodeFeatures.targetEncoding(X_train, Y_data, X_test, col, group_by_features, enc_stats=None)
X_train.drop(['price'], axis=1, inplace=True)


print('\n=============================>  ɾ����������\n')
time_features_new = []
for f, t in zip(X_data.dtypes.index, X_data.dtypes.values):
    ## ���˳�ʱ������
    if str(t)[:8] == 'datetime':
        time_features_new.append(f)
drop_list = ['regDate', 'creatDate', 'brand_power_min', 'regDate_year_power_min'] + time_features_new
x_train = X_train.drop(drop_list, axis=1)
x_test = X_test.drop(drop_list, axis=1)

print('\n=============================>  ����NaN��Inf\n')
num_train = x_train.shape[0]
concat_data = pd.concat([x_train, x_test])
columns = concat_data.columns.tolist()
num_inf_before = np.isinf(concat_data.values).sum()
num_nan_before = concat_data.isnull().sum().sum()
if (num_inf_before != 0) | (num_nan_before != 0):
    print('����ǰ INF:{},NaN:{}'.format(num_inf_before, num_nan_before))
    concat_data = dataPreprocess.replace_var(concat_data, cols_name=columns)
    num_inf_after = np.isinf(concat_data.values).sum()
    num_nan_after = concat_data.isnull().sum().sum()
    print('����� INF:{},NaN:{}'.format(num_inf_after, num_nan_after))
x_train = concat_data.iloc[:num_train, :]
x_test = concat_data.iloc[num_train:, :]

for f in x_train.columns:
    print(f)


analysis_data = x_train.copy()
analysis_data['price'] = Y_data['price']
###��������
Wrapper_models = [ LogisticRegression(), LGBMRegressor()]
Embedded_models = [LogisticRegression(), LGBMRegressor()]
getScores = ScoreFeatures.Score_of_features(analysis_data, x_train.columns, 'price', 'neg_mean_absolute_error')
score_df = getScores.score_main(Wrapper_models=Wrapper_models, Embedded_models=Embedded_models)

# getScores.figs_of_RFE(model=LogisticRegression())
# getScores.figs_of_SFS(model=LogisticRegression())

mean_s = score_df.loc[:, 'mean score'].sort_index(axis=0, ascending=False, by=['mean score'])
print(mean_s.head(20))

median_s = score_df.loc[:, 'median score'].sort_index(axis=0, ascending=False, by=['median score'])
print(median_s.head(20))

max_s = score_df.loc[:, 'max score'].sort_index(axis=0, ascending=False, by=['max score'])
print(max_s.head(20))




print('\n=============================>  PCA��ά\n')
print('ѹ��ǰ��������{}'.format(x_train.shape[1]))
pca_dim = 200
if pca_dim > x_train.shape[1]:
    pca_dim = x_train.shape[1]
X_train_pca, X_test_pca = ReduceDimension.dimen_reduct_by_pca(x_train, x_test, pca_dim=pca_dim)
print('�ӳ�ʼ{}������ѹ����{}������'.format(x_train.shape[1],X_train_pca.shape[1]))
print('����ѵ������С��', X_train_pca.shape)
print('���ղ��Լ���С��', X_test_pca.shape)

print('\n=============================>  ��������\n')
X_train_df = pd.DataFrame(X_train_pca)
X_test_df = pd.DataFrame(X_test_pca)
X_train_df['price'] = Y_data['price']

X_train_path = config.path_data.dataOriPath + r'/used_car_{}Features_train.csv'.format(X_test_pca.shape[1])
X_test_path = config.path_data.dataOriPath + r'/used_car_{}Features_test.csv'.format(X_test_pca.shape[1])

X_train_df.to_csv(X_train_path)
X_test_df.to_csv(X_test_path)


# aa = pd.read_csv(X_train_path, index_col=0)
# bb = pd.read_csv(X_test_path, index_col=0)
# print(aa.shape)
# print(bb.shape)
# print(aa.head())
# print(bb.head())