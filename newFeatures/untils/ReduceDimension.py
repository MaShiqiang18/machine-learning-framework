# coding=gbk

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition

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