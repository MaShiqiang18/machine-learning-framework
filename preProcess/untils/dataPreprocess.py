# coding=gbk

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer as Imputer
from sklearn.model_selection import KFold
from tqdm import tqdm
import re

#处理异常值
def outliers_proc(data, col_name, scale=3, comp=False, show_view=False):
    """
    用于清洗异常值，默认用 box_plot（scale=3）进行清洗
    :param data: 接收 pandas 数据格式
    :param col_name: pandas 列名
    :param scale: 尺度
    :param comp: 是否显示原数据和异常数据分布对比
    :param show_view: 是否显示删除异常值前后的箱型图对比
    :return: 异常的样本的index
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度，
        :return:
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_series = data_n[col_name]
    rule, value = box_plot_outliers(data_series, box_scale=scale)
    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
    print("有{}个异常样本".format(len(index)))
    # data_n = data_n.drop(index)
    ##
    data_n.reset_index(drop=True, inplace=True)
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]

    if comp:
        print("删除异常值后的样本数：{}".format(data_n.shape[0]))
        print("小于阈值的数据分布:")
        print(pd.Series(outliers).describe())
        print("大于阈值的数据分布:")
        print(pd.Series(outliers).describe())

    if show_view:
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
        sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
        plt.show()
    return index

# 处理异常值
def smooth_cols(data,cols,out_value,kind):
    """
    给定阈值，将超出阈值部分直接用设定的分位值替换
    :param data: 数据集
    :param cols: 指定特征
    :param out_value: 阈值
    :param kind: 类型：上限&下限
    :return: 替换后的数据集
    """

    if kind == 'g':
        for col in cols:
            yes_no = (data[col]<out_value).astype('int')
            new = yes_no * data[col]
            data[col] = new.replace(0,data[col].quantile(q=0.995))
        return data
    if kind == 'l':
        for col in cols:
            yes_no = (data[col]>out_value).astype('int')
            new = yes_no * data[col]
            data[col] = new.replace(0,data[col].quantile(q=0.07))
        return data

# 处理缺失值
def replace_var_obj(data,missing_values='NaN', strategy='mean', axis=0):
    """
    :param data: 需要替换值的元数据
    :param missing_values: 需替换的值，默认为缺失值NaN
    :param strategy: 替换的方式，可选：'mean','median','most_frequent'
    :param axis: 按行或列替换
    :return: 替换后的数据
    """

    if len(data.shape) == 1:
        num_all_data = len(data)
        if axis == 0:
            data = np.array(data).reshape(num_all_data, 1)
        else:
            data = np.array(data).reshape(1,num_all_data)
    else:
        data = np.array(data)

    imr_mean = Imputer(strategy=strategy)  # 均值填充缺失值
    imr_mean_0 = imr_mean.fit(data)
    imputed_data = imr_mean.transform(data)
    return imputed_data

# 处理缺失值
def replace_var(data,cols_name=None):
    '''
    对DataFrame类型的数据data中的cols_name这几列作缺失值替换,只处理数值类型和字符串类型
    :param data: 需要替换的数据
    :param cols_name: 指定的需处理的列
    :return: 替换后的数据
    '''
    go_on = True
    if cols_name is None:
        # 若不知道处理特征，默认处理全部特征
        cols_name = data.columns.values.tolist()
        print('请确认需要全部的特征进行缺失值处理？若默认全部特征请输入：y')
        use_all = input('请输入：')
        if use_all == 'y':
            go_on = True
        else:
            go_on = False
            print('请指定特征后，重新处理')

    if go_on:
        desc_all = data[cols_name].describe()
        num_all_data = data.shape[0]

        for c in cols_name:
            # 只处理数值特征
            if str(data[c].dtype) in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64']:
                data_c = data[c].copy()
                # inf(无穷大)转化为NaN
                data_c[np.isinf(data_c.values)] = np.nan
                # 缺失率
                P_null = data_c.isnull().sum()/num_all_data

                if P_null != 0.0:
                    if P_null > 0.3:
                        # 缺失值大于0.3，直接删除列
                        data.drop(columns=c, inplace=True)
                        print('特征：{}，缺失率大于0.3，已删除'.format(c))
                    else:
                        # 众数占比
                        P_most_var = data_c.value_counts(normalize=True, dropna=False).values[0] * 100
                        if P_most_var > 0.6:
                            # 当某一个特征值占比超过60%,众数填充缺失值
                            imputed_data = replace_var_obj(data_c, missing_values='NaN', strategy='most_frequent', axis=0)
                        else:
                            # 数据分析，是否满足正态分布，极大极小值是否合理
                            mean_c = desc_all[c]['mean']
                            std_c = desc_all[c]['std']
                            min_c = desc_all[c]['min']
                            max_c = desc_all[c]['max']
                            normal_distribution = (min_c < mean_c-3*std_c**2) and (max_c < mean_c+3*std_c**2)
                            if normal_distribution:
                                # 均值填充缺失值
                                imputed_data = replace_var_obj(data_c, missing_values='NaN', strategy='mean', axis=0)
                            else:
                                # 中位数填充缺失值
                                imputed_data = replace_var_obj(data_c, missing_values='NaN', strategy='median', axis=0)
                        data[c] = imputed_data
            else:
                P_null = data[c].isnull().sum() / num_all_data
                if P_null != 0.0:
                    print('\n**************特征'+c+'的类型为'+str(data[c].dtype)+',单独处理**************\n')
    return data

# 处理数据类型
def date_proc(x):
    '''
    整型的时间特征转化为时间戳格式
    :param x:格式为：20200305
    :return:
    '''
    if len(x) != 8:
        print('数据：{}格式不合'.format(x))
        return None
    else:
        m = int(x[4:6])
        if m == 0:
            m = 1
        return x[:4] + '-' + str(m) + '-' + x[6:]


def reduce_mem_usage(df, time_features=None):
    """
    通过调整数据类型，帮助我们减少数据在内存中占用的空间
    """

    if not time_features:
        print('有无时间特征请先提前处理？若无请输入：no，若有请输入特征名，list用“，”隔开，用“\\n”结束')
        time_features = input('请输入：')
        if time_features == 'no':
            other_features = df.columns
        else:
            sp_features = time_features.strip().split(',')
            other_features = df.columns.tolist()
            for f in sp_features:
                if f not in other_features:
                    print('不存在时间特征：{}，请确认！'.format(f))
                else:
                    other_features.remove(f)
                    df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
    else:
        sp_features = time_features
        other_features = df.columns.tolist()
        for f in sp_features:
            if f not in other_features:
                print('不存在时间特征：{}，请确认！'.format(f))
            else:
                other_features.remove(f)
                df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
                a = df[f].dtypes


    start_mem = df.memory_usage().sum()/1024/1024
    print('原先所需内存 {:.2f} MB'.format(start_mem))

    for col in other_features:
        col_type = df[col].dtype
        if col_type != object:
            if str(col_type)[:3] == 'int':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            elif str(col_type)[:5] == 'float':
                c_min = df[col].min()
                c_max = df[col].max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            elif str(col_type)[:4] == 'bool':
                df[col] = df[col].astype(np.int8)
            else:
                print('将特征：{}，从原数据类型：{}转化为category数据类型'.format(col, str(col_type)))
                df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()/1024/1024
    print('处理后所需内存: {:.2f} MB'.format(end_mem))
    print('内存节省率 {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


