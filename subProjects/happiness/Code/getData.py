# coding=gbk
import pandas as pd
from subProjects.happiness.Code.config import ParamConfig

mark = 'v1'
mark_text = ''
config = ParamConfig(mark=mark, mark_text=mark_text)
print('\n=============================>  准备数据集\n')
train_data = pd.read_csv(config.path_data.dataOriPath+'/happiness_train_complete.csv',sep=',',encoding='latin-1')
test_data = pd.read_csv(config.path_data.dataOriPath+'/happiness_test_complete.csv',sep=',',encoding='latin-1')
submit_example = pd.read_csv(config.path_data.dataOriPath+'/happiness_submit.csv',sep=',',encoding='latin-1')
columns = train_data.columns.tolist()
label = config.data_label

print('train shape:',train_data.shape)
print('test shape:',test_data.shape)
print('sample shape:',submit_example.shape)
print(columns)

from analysis.main import run_analysis
from analysis.untils import dataAnalysis

birth_features = ['birth', 's_birth', 'f_birth',  'm_birth']
time_features = ['survey_time',  'join_party', 'marital_1st', 'marital_now']
income = ['income', 'family_income', 'inc_exp', 's_income']
family = ['family_m', 'son', 'daughter', 'minor_child', ]
num_features = ['religion_freq', 'floor_area', 'height_cm', 'weight_jin', 'work_yr']
public_service = ['public_service_1', 'public_service_2', 'public_service_3', 'public_service_4', 'public_service_5',
                  'public_service_6', 'public_service_7', 'public_service_8', 'public_service_9']

# categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox']
# run_analysis.analysis_main(train_data, columns, label=label, categorical_features=categorical_features, num_features=num_features)

analysis_all_features = dataAnalysis.Analysis_all_features(train_data,columns,label)
analysis_all_features.show_shape()
analysis_all_features.percentage_miss()
analysis_all_features.show_corr_label_with_features(show_picture=False)
