# coding=gbk

import pandas as pd
import numpy as np
from newFeatures.untils.MeanEncoder import MeanEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition
from sklearn.model_selection import KFold
from tqdm import tqdm


"""
用于训练集、测试集分别构建新特征（编码特征）：
1. 训练集先编码
2. 测试集依照训练集编码
"""

def meanEnocoding(X_data,Y_data,X_test, Feature_list):
    """
    Feature_list 特征值满足下面几点：1. 会重复，2. 根据相同的值分组会分出超过一定数量（比如100）的组.
    """
    # 声明需要平均数编码的特征
    MeanEnocodeFeature = Feature_list
    # 声明平均数编码的类
    ME = MeanEncoder(MeanEnocodeFeature, target_type='regression')
    # 对训练数据集的X和y进行拟合
    X_train = ME.fit_transform(X_data, Y_data)
    # 对训练数据集的X和y进行拟合
    #x_train_fav = ME.fit_transform(x_train,y_train_fav)
    # 对测试集进行编码
    X_test = ME.transform(X_test)

    return X_train, X_test




### target encoding目标编码，回归场景相对来说做目标编码的选择更多，不仅可以做均值编码，还可以做标准差编码、中位数编码等
def targetEncoding(X_data,Y_data,X_test, col, group_by_features, enc_stats=None):
    """
    拆分为10折后，每一折按group_by_features中的一个特征分组，
    取出分组后每一组的col的最大、最小、均值（按enc_stats定）作为新特征
    :param X_data:
    :param Y_data:
    :param X_test:
    :param col: 数值特征
    :param group_by_features: 类别特征
    :param enc_stats: stats_default_dict中的keys任选
    :return:
    """
    enc_cols = []
    stats_default_dict = {
        'max': X_data[col].max(),
        'min': X_data[col].min(),
        'median': X_data[col].median(),
        'mean': X_data[col].mean(),
        'sum': X_data[col].sum(),
        'std': X_data[col].std(),
        'skew': X_data[col].skew(),
        'kurt': X_data[col].kurt(),
        'mad': X_data[col].mad()
    }
    ### 暂且选择这三种编码
    if not enc_stats:
        enc_stats = ['max','min','mean']
    skf = KFold(n_splits=10, shuffle=True, random_state=42)
    for f in tqdm(group_by_features):
        enc_dict = {}
        for stat in enc_stats:
            enc_dict['{}_target_{}_{}'.format(f, col, stat)] = stat
            X_data['{}_target_{}_{}'.format(f, col, stat)] = 0
            X_test['{}_target_{}_{}'.format(f, col, stat)] = 0
            enc_cols.append('{}_target_{}_{}'.format(f, col, stat))
        for i, (trn_idx, val_idx) in enumerate(skf.split(X_data, Y_data)):
            trn_x, val_x = X_data.iloc[trn_idx].reset_index(drop=True), X_data.iloc[val_idx].reset_index(drop=True)
            enc_df = trn_x.groupby(f, as_index=False)[col].agg(enc_dict)
            val_x = val_x[[f]].merge(enc_df, on=f, how='left')
            test_x = X_test[[f]].merge(enc_df, on=f, how='left')
            for stat in enc_stats:
                val_x['{}_target_{}_{}'.format(f, col, stat)] = val_x['{}_target_{}_{}'.format(f, col, stat)].fillna(stats_default_dict[stat])
                test_x['{}_target_{}_{}'.format(f, col, stat)] = test_x['{}_target_{}_{}'.format(f, col, stat)].fillna(stats_default_dict[stat])
                X_data.loc[val_idx, '{}_target_{}_{}'.format(f, col, stat)] = val_x['{}_target_{}_{}'.format(f, col, stat)].values
                X_test['{}_target_{}_{}'.format(f, col, stat)] += test_x['{}_target_{}_{}'.format(f, col, stat)].values / skf.n_splits

    return X_data, X_test


#特征降维
def dimen_reduct_by_pca(x_train, x_test, pca_dim=10):
    if pca_dim >= x_train.shape[1]:
        print('指定维数小于初始特征数，不做降维操作！')
        return x_train, x_test

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    train_num_null = x_train.isnull().sum()
    test_num_null = x_test.isnull().sum()
    if train_num_null.sum() != 0:
        print('请先确保训练集中没有空值')
        if train_num_null.sum() != 0:
            for f in x_train.columns:
                if train_num_null[f] != 0:
                    print('特征{}有空值：{}'.format(f, train_num_null[f]))
        return x_train, x_test
    elif test_num_null.sum() != 0:
        print('请先确保测试集中没有空值')
        if test_num_null.sum() != 0:
            for f in x_test.columns:
                if test_num_null[f] != 0:
                    print('特征{}有空值：{}'.format(f, test_num_null[f]))
        return x_train, x_test
    else:
        if x_train.shape[1] != x_test.shape[1]:
            print('请确保训练集和测试集的特征数一致！')
            print('现在训练集特征有：{}，测试集特征有：{}'.format(x_train.shape[1], x_test.shape[1]))
            if x_train.shape[1] > x_test.shape[1]:
                the_features = [f for f in x_train.columns if f not in x_test.columns]
                print('训练集中存在，但是测试集中不存在的特征有：')
                print(the_features)
            else:
                the_features = [f for f in x_test.columns if f not in x_train.columns]
                print('测试集中存在，但是训练集中不存在的特征有：')
                print(the_features)
            return x_train, x_test
        min_max_scaler = MinMaxScaler()
        min_max_scaler.fit(pd.concat([x_train, x_test]).values)
        all_data = min_max_scaler.transform(pd.concat([x_train, x_test]).values)

        pca = decomposition.PCA(n_components=pca_dim)
        all_pca = pca.fit_transform(all_data)
        X_train_pca = all_pca[:len(x_train)]
        X_test_pca = all_pca[len(x_train):]

        return X_train_pca, X_test_pca

