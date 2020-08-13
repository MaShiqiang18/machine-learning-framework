# coding=gbk

import pandas as pd
from subProjects.used_car.Code.config import ParamConfig

mark = 'v1'
mark_text = ''
config = ParamConfig(mark=mark, mark_text=mark_text)

train_path = config.path_data.dataOriPath + r'/used_car_train_20200313.csv'
test_path = config.path_data.dataOriPath + r'/used_car_testB_20200421.csv'
train = pd.read_csv(train_path, sep=' ', encoding='gb18030')
test = pd.read_csv(test_path, sep=' ', encoding='gb18030')
target_col = config.data_label

v0 = []
v00 = train.loc[(train['v_0']<44) & (train[target_col]>60000),:].index.tolist()
v0.extend(v00)

v10 = train.loc[(train['v_1']>0.5) & (train['v_1']<1) & (train[target_col]>60000),:].index.tolist()
v0.extend(v10)

v20 = train.loc[(train['v_2']<-2) & (train[target_col]>40000) | (train['v_2']>15) & (train[target_col]>60000),:].index.tolist()
v0.extend(v20)

v30 = train.loc[train['v_3']<-7.0,:].index.tolist()
v0.extend(v30)

v40 = train.loc[train['v_4']>6,:].index.tolist()
v0.extend(v40)

v50 = train.loc[(train['v_5']<0.2) & (train[target_col]>45000),:].index.tolist()
v0.extend(v50)

# v60 = train.loc[(train['v_6']>0.04) & (train['v_6']<0.06) & (train[target_col]>60000),:].index.tolist()
# v0.extend(v60)

v70 = train.loc[(train['v_7']>1.0) & (train[target_col]>60000),:].index.tolist()
v0.extend(v70)

v80 = train.loc[(train['v_8']<0.07) & (train[target_col]>80000),:].index.tolist()
v0.extend(v80)

v90 = train.loc[(train['v_9']>0.2) & (train[target_col]>40000),:].index.tolist()
v0.extend(v90)

v100 = train.loc[(train['v_10']>7) & (train[target_col]>60000),:].index.tolist()
v0.extend(v100)

v110 = train.loc[(train['v_11']>2) & (train[target_col]>60000),:].index.tolist()
v0.extend(v110)

v120 = train.loc[(train['v_12']>12) & (train[target_col]>60000),:].index.tolist()
v0.extend(v120)

v130 = train.loc[train['v_13']>10,:].index.tolist()
v0.extend(v130)

v140 = train.loc[train['v_14']>8,:].index.tolist()
v0.extend(v140)

drop_index = set(v0)
abnormal = train.loc[drop_index, :]
print('异常点：', len(abnormal))
print(abnormal)
abnormal.to_csv(config.path_data.dataOriPath + r'/abnormal_index.csv', header=True, index=True)