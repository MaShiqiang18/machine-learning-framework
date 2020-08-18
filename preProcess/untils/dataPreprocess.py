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

#�����쳣ֵ
def outliers_proc(data, col_name, scale=3, comp=False, show_view=False):
    """
    ������ϴ�쳣ֵ��Ĭ���� box_plot��scale=3��������ϴ
    :param data: ���� pandas ���ݸ�ʽ
    :param col_name: pandas ����
    :param scale: �߶�
    :param comp: �Ƿ���ʾԭ���ݺ��쳣���ݷֲ��Ա�
    :param show_view: �Ƿ���ʾɾ���쳣ֵǰ�������ͼ�Ա�
    :return: �쳣��������index
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        ��������ͼȥ���쳣ֵ
        :param data_ser: ���� pandas.Series ���ݸ�ʽ
        :param box_scale: ����ͼ�߶ȣ�
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
    print("��{}���쳣����".format(len(index)))
    # data_n = data_n.drop(index)
    ##
    data_n.reset_index(drop=True, inplace=True)
    index_low = np.arange(data_series.shape[0])[rule[0]]
    outliers = data_series.iloc[index_low]
    index_up = np.arange(data_series.shape[0])[rule[1]]
    outliers = data_series.iloc[index_up]

    if comp:
        print("ɾ���쳣ֵ�����������{}".format(data_n.shape[0]))
        print("С����ֵ�����ݷֲ�:")
        print(pd.Series(outliers).describe())
        print("������ֵ�����ݷֲ�:")
        print(pd.Series(outliers).describe())

    if show_view:
        fig, ax = plt.subplots(1, 2, figsize=(10, 7))
        sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
        sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
        plt.show()
    return index

# �����쳣ֵ
def smooth_cols(data,cols,out_value,kind):
    """
    ������ֵ����������ֵ����ֱ�����趨�ķ�λֵ�滻
    :param data: ���ݼ�
    :param cols: ָ������
    :param out_value: ��ֵ
    :param kind: ���ͣ�����&����
    :return: �滻������ݼ�
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

# ����ȱʧֵ
def replace_var_obj(data,missing_values='NaN', strategy='mean', axis=0):
    """
    :param data: ��Ҫ�滻ֵ��Ԫ����
    :param missing_values: ���滻��ֵ��Ĭ��ΪȱʧֵNaN
    :param strategy: �滻�ķ�ʽ����ѡ��'mean','median','most_frequent'
    :param axis: ���л����滻
    :return: �滻�������
    """

    if len(data.shape) == 1:
        num_all_data = len(data)
        if axis == 0:
            data = np.array(data).reshape(num_all_data, 1)
        else:
            data = np.array(data).reshape(1,num_all_data)
    else:
        data = np.array(data)

    imr_mean = Imputer(strategy=strategy)  # ��ֵ���ȱʧֵ
    imr_mean_0 = imr_mean.fit(data)
    imputed_data = imr_mean.transform(data)
    return imputed_data

# ����ȱʧֵ
def replace_var(data,cols_name=None):
    '''
    ��DataFrame���͵�����data�е�cols_name�⼸����ȱʧֵ�滻,ֻ������ֵ���ͺ��ַ�������
    :param data: ��Ҫ�滻������
    :param cols_name: ָ�����账�����
    :return: �滻�������
    '''
    go_on = True
    if cols_name is None:
        # ����֪������������Ĭ�ϴ���ȫ������
        cols_name = data.columns.values.tolist()
        print('��ȷ����Ҫȫ������������ȱʧֵ������Ĭ��ȫ�����������룺y')
        use_all = input('�����룺')
        if use_all == 'y':
            go_on = True
        else:
            go_on = False
            print('��ָ�����������´���')

    if go_on:
        desc_all = data[cols_name].describe()
        num_all_data = data.shape[0]

        for c in cols_name:
            # ֻ������ֵ����
            if str(data[c].dtype) in ['int8', 'int16', 'int32', 'int64', 'float8', 'float16', 'float32', 'float64']:
                data_c = data[c].copy()
                # inf(�����)ת��ΪNaN
                data_c[np.isinf(data_c.values)] = np.nan
                # ȱʧ��
                P_null = data_c.isnull().sum()/num_all_data

                if P_null != 0.0:
                    if P_null > 0.3:
                        # ȱʧֵ����0.3��ֱ��ɾ����
                        data.drop(columns=c, inplace=True)
                        print('������{}��ȱʧ�ʴ���0.3����ɾ��'.format(c))
                    else:
                        # ����ռ��
                        P_most_var = data_c.value_counts(normalize=True, dropna=False).values[0] * 100
                        if P_most_var > 0.6:
                            # ��ĳһ������ֵռ�ȳ���60%,�������ȱʧֵ
                            imputed_data = replace_var_obj(data_c, missing_values='NaN', strategy='most_frequent', axis=0)
                        else:
                            # ���ݷ������Ƿ�������̬�ֲ�������Сֵ�Ƿ����
                            mean_c = desc_all[c]['mean']
                            std_c = desc_all[c]['std']
                            min_c = desc_all[c]['min']
                            max_c = desc_all[c]['max']
                            normal_distribution = (min_c < mean_c-3*std_c**2) and (max_c < mean_c+3*std_c**2)
                            if normal_distribution:
                                # ��ֵ���ȱʧֵ
                                imputed_data = replace_var_obj(data_c, missing_values='NaN', strategy='mean', axis=0)
                            else:
                                # ��λ�����ȱʧֵ
                                imputed_data = replace_var_obj(data_c, missing_values='NaN', strategy='median', axis=0)
                        data[c] = imputed_data
            else:
                P_null = data[c].isnull().sum() / num_all_data
                if P_null != 0.0:
                    print('\n**************����'+c+'������Ϊ'+str(data[c].dtype)+',��������**************\n')
    return data

# ������������
def date_proc(x):
    '''
    ���͵�ʱ������ת��Ϊʱ�����ʽ
    :param x:��ʽΪ��20200305
    :return:
    '''
    if len(x) != 8:
        print('���ݣ�{}��ʽ����'.format(x))
        return None
    else:
        m = int(x[4:6])
        if m == 0:
            m = 1
        return x[:4] + '-' + str(m) + '-' + x[6:]


def reduce_mem_usage(df, time_features=None):
    """
    ͨ�������������ͣ��������Ǽ����������ڴ���ռ�õĿռ�
    """

    if not time_features:
        print('����ʱ������������ǰ�������������룺no��������������������list�á������������á�\\n������')
        time_features = input('�����룺')
        if time_features == 'no':
            other_features = df.columns
        else:
            sp_features = time_features.strip().split(',')
            other_features = df.columns.tolist()
            for f in sp_features:
                if f not in other_features:
                    print('������ʱ��������{}����ȷ�ϣ�'.format(f))
                else:
                    other_features.remove(f)
                    df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
    else:
        sp_features = time_features
        other_features = df.columns.tolist()
        for f in sp_features:
            if f not in other_features:
                print('������ʱ��������{}����ȷ�ϣ�'.format(f))
            else:
                other_features.remove(f)
                df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
                a = df[f].dtypes


    start_mem = df.memory_usage().sum()/1024/1024
    print('ԭ�������ڴ� {:.2f} MB'.format(start_mem))

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
                print('��������{}����ԭ�������ͣ�{}ת��Ϊcategory��������'.format(col, str(col_type)))
                df[col] = df[col].astype('category')
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()/1024/1024
    print('����������ڴ�: {:.2f} MB'.format(end_mem))
    print('�ڴ��ʡ�� {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


