# coding=gbk

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import decomposition

#������ά
def dimen_reduct_by_pca(x_train, x_test, pca_dim=10):
    if pca_dim >= x_train.shape[1]:
        print('ָ��ά��С�ڳ�ʼ��������������ά������')
        return x_train, x_test

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    train_num_null = x_train.isnull().sum()
    test_num_null = x_test.isnull().sum()
    if train_num_null.sum() != 0:
        print('����ȷ��ѵ������û�п�ֵ')
        if train_num_null.sum() != 0:
            for f in x_train.columns:
                if train_num_null[f] != 0:
                    print('����{}�п�ֵ��{}'.format(f, train_num_null[f]))
        return x_train, x_test
    elif test_num_null.sum() != 0:
        print('����ȷ�����Լ���û�п�ֵ')
        if test_num_null.sum() != 0:
            for f in x_test.columns:
                if test_num_null[f] != 0:
                    print('����{}�п�ֵ��{}'.format(f, test_num_null[f]))
        return x_train, x_test
    else:
        if x_train.shape[1] != x_test.shape[1]:
            print('��ȷ��ѵ�����Ͳ��Լ���������һ�£�')
            print('����ѵ���������У�{}�����Լ������У�{}'.format(x_train.shape[1], x_test.shape[1]))
            if x_train.shape[1] > x_test.shape[1]:
                the_features = [f for f in x_train.columns if f not in x_test.columns]
                print('ѵ�����д��ڣ����ǲ��Լ��в����ڵ������У�')
                print(the_features)
            else:
                the_features = [f for f in x_test.columns if f not in x_train.columns]
                print('���Լ��д��ڣ�����ѵ�����в����ڵ������У�')
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