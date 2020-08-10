# coding=gbk

from preProcess.untils import dataPreprocess

"""
数据分析后，特征工程前，进行数据预处理
主要用于：异常值处理、空值处理、数值类型转化（时间特征）
"""


def preprocess_main(data_ori, categorical_features=None, num_features=None, time_features=None, fillnan=True):
    """
    数据预处理，主要用于异常值处理(识别异常值，不删除)、空值处理、数据类型转化
    :param data_ori: 原始数据
    :param columns: 所有特征名
    :param labels: 标签名
    :param categorical_features: 类别特征
    :param num_features: 数值特征
    :param time_features: 时间特征
    :param fillnan: 是否填充缺失值
    :return:
    """
    data = data_ori.copy()
    columns = data.columns.tolist()

    print('\n\n******************处理数值特征的异常值******************')
    abnormal_index = []
    for f in num_features:
        print('特征：'+f)
        index_f = dataPreprocess.outliers_proc(data, f, scale=3, comp=False, show_view=False)
        abnormal_index.extend(index_f.tolist())
    abnormal_index_set = set(abnormal_index)
    print('\n====>识别出总共{}个样本'.format(len(abnormal_index_set)))


    num_null = data.isnull().sum().sum()
    if (num_null > 0) & fillnan:
        print('\n\n************************处理空值************************')
        ### 特殊空值（-）的处理，收到指定代替
        # sp_feature = ''
        # concat_data[sp_feature] = concat_data[sp_feature].replace('-', 0).astype('float16')
        data = dataPreprocess.replace_var(data, cols_name=columns)
        num_null = data.isnull().sum()
        if num_null.sum() != 0:
            for f in columns:
                if num_null[f] != 0:
                    print('特征{}有空值：{}'.format(f, num_null[f]))
        else:
            print('空值填充完毕！')

    print('\n\n**********************转化数据类型**********************')
    data = dataPreprocess.reduce_mem_usage(data, time_features)

    return data, abnormal_index_set
