# coding=gbk

import pandas as pd
import numpy as np
import datetime
from scipy.stats import entropy
from tqdm import tqdm
from datetime import datetime
from preProcess.untils.dataPreprocess import date_proc
import re


"""
����ѵ���������Լ��ϲ���һ�𹹽�������
"""

######################################################ʱ������######################################################
#������ȡ
def date_get_ymdw(df, fea_col, add_new=True):
    print('��ȡʱ��������{}����ȡ�ꡢ�¡��ա��ܵ���Ϣ'.format(fea_col))
    features_ori = df.columns
    for f in tqdm(fea_col):
        if str(df[f].dtypes)[:8] != 'datetime':
            df[f] = pd.to_datetime(df[f].astype('str').apply(date_proc))
        df[f + '_year'] = df[f].dt.year
        df[f + '_month'] = df[f].dt.month
        df[f + '_day'] = df[f].dt.day
        df[f + '_dayofweek'] = df[f].dt.dayofweek

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new

# ����ʱ������
def date_features(df, f1, f2, add_new=True):
    """
    f1-f2�����ڲ��ǰ������f1��ʱ����ǰ������f2�����ڲ�
    :param df:
    :param f1:
    :param f2:
    :return:
    """
    print('��ȡʱ��������{}��{}��ʱ���'.format(f1, f2))
    features_ori = df.columns
    if str(df[f1].dtypes)[:8] != 'datetime':
        df[f1] = pd.to_datetime(df[f1].astype('str').apply(date_proc))
    if str(df[f2].dtypes)[:8] != 'datetime':
        df[f2] = pd.to_datetime(df[f2].astype('str').apply(date_proc))

    df[f1+'_sub_'+f2 + '_days'] = (df[f1] - df[f2]).dt.days
    df['now_sub_'+f1 + '_days'] = (datetime.now() - df[f2]).dt.days
    df['now_sub_'+f2 + '_days'] = (datetime.now() - df[f1]).dt.days
    # df[f1+'_'+f2] = (pd.to_datetime(df[f1], format='%Y%m%d', errors='coerce') -
    #                       pd.to_datetime(df[f2], format='%Y%m%d', errors='coerce')).dt.days
    # df['now_'+f1] = (pd.datetime.now() - pd.to_datetime(df[f2], format='%Y%m%d', errors='coerce')).dt.days
    # df['now_'+f2] = (pd.datetime.now() - pd.to_datetime(df[f1], format='%Y%m%d', errors='coerce')).dt.days

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new



######################################################��ֵ����######################################################
#��������ͳ����
import math
def produce_by_single(data, num_cols):
    print('��ȡ������{}ƽ������������������Ϣ'.format(num_cols))
    for i in num_cols:
        data[i + '**2'] = data[i].map(lambda x: x**2)

    for i in num_cols:
        data[i + '**3'] = data[i].map(lambda x: x**3)

    # for i in num_cols:
    #     # data['log_' + i] = data[i].map(lambda x: math.log(x+1))
    #     data['log_' + i] = np.log1p(data[i])
    return data

def produce_by_double(data, num_cols):
    print('��ȡ������{}�໥����Ӽ����������Ϣ'.format(num_cols))
    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            data[num_cols[i] + '*' + num_cols[j]] = data[num_cols[i]].mul(data[num_cols[j]], fill_value=0)

    for i in range(len(num_cols)):
        for j in range(i+1, len(num_cols)):
            data[num_cols[i] + '+' + num_cols[j]] = data[num_cols[i]].add(data[num_cols[j]], fill_value=0)

    for i in num_cols:
        for j in num_cols:
            if i != j:
                data[i + '-' + j] = data[i].sub(data[j], fill_value=0)

    for i in num_cols:
        for j in num_cols:
            if i != j:
                div_values = data[i].div(data[j], fill_value=0)
                data[i + '/' + j] = data[i].div(data[j], fill_value=0)

    for i in num_cols:
        for j in num_cols:
            if i != j:
                data[i + '- ' + j + '/' + i] = data[i + '- ' + j].div(data[j], fill_value=0)

    return data


#��Ͱ����
def cut_group(df, cols, num_bins=50, add_new=True):
    print('��������{}���з�Ͱ��{}������'.format(cols, num_bins))
    features_ori = df.columns
    for col in cols:
        all_range = int(df[col].max()-df[col].min())
        bin = [i*all_range/num_bins for i in range(all_range)]
        df[col+'_bin'] = pd.cut(df[col], bin, labels=False)

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new



######################################################�������######################################################
### count����
def count_coding(df, fea_col, add_new=True):
    print('��������{}����count����'.format(fea_col))
    features_ori = df.columns
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new

# ��ȡonehot����
def gen_onehot(df, col_name='F', deal_with_sp=False):
    """
    ��df������col_name���б���
    �Աȣ�pd.get_dummies(data, columns=col_names)
    ��ͬ�㣺gen_onehot�ɴ���������ַ�
    example:
            a1 = pd.Series(['��','@','c'])
            b1 = pd.Series([1,2,3])
            c1 = pd.DataFrame({'F1':a1,'F2':b1})

            print(c1)
            r = gen_onehot(c1,col_name='F1')
            print(r)
            t = pd.get_dummies(c1, columns=['F1','F2'])
            print(t)

            output:

                  F1  F2
                0  ��   1
                1  @   2
                2  c   3
                   F1_NaN  F1_c  F1_��
                0       0     0     1
                1       1     0     0
                2       0     1     0
                   F1_@  F1_c  F1_��  F2_1  F2_2  F2_3
                0     0     0     1     1     0     0
                1     1     0     0     0     1     0
                2     0     1     0     0     0     1
    """

    if col_name not in df.columns:
        print('��ȷ������{}�ڸ����ݼ���'.format(col_name))
        return df
    else:
        data_df = df
        data_with_new_features = pd.get_dummies(data_df[col_name])

        if deal_with_sp:
            # ������������
            new_cols_names = []
            cols = 0
            for col in data_with_new_features.columns.values.tolist():
                str_sub = re.sub("[\#\@\$\&\*\(\)\\\~\`\<\>\.\-\_\!\%\[\]\,\��]", "", col)
                if str_sub != "":
                    new_cols_names.append(col_name + '_'+col)
                else:
                    if cols == 0:
                        new_cols_names.append(col_name + '_NaN')
                    else:
                        new_cols_names.append(col_name + '_' + str(cols))
                    cols += 1
            data_with_new_features.columns = new_cols_names

        return data_with_new_features

# ��onehot����ȡ��ԭ����
def add_onehot_features(data_ori, cols_to_onehot, add_new=True):
    """
    ��Onehot����������ȡ��������
    :param data_ori:
    :param cols_to_onehot:list
    :return:
    """
    print('��������{}����onehot����'.format(cols_to_onehot))
    features_ori = data_ori.columns
    data = data_ori.copy()
    for c in cols_to_onehot:
        print('\n��������{}����onehot���롣����'.format(c))
        data_with_new_features = gen_onehot(data, col_name=c)
        print('����onehot����������')
        print(data_with_new_features.columns.values.tolist())
        print('\n')
        # �ñ�������ȡ��ԭ����
        data = data.drop(columns=[c], axis=1)
        data = pd.concat([data, data_with_new_features], axis=1)


    if add_new:
        return data
    else:
        features_new = [f for f in data.columns if f not in features_ori]
        df_new = data.loc[:, features_new]
        return df_new



######################################################��������######################################################
def produce_by_statistics(data, num_col, cat_col, add_new=True):
    """
    ������������ͳ������Ϊ������
    :param data: dataFrame��ʽ
    :param col_1: ���ڷ��������
    :param col_2: ����ͳ�Ƶ�����
    :return: brand_fe����������data���ϲ�������������ݼ�
    """
    print('��������{}������������{}�����'
          'ͳ�����������ֵ����Сֵ����ֵ���ܺ͡���ֵ�����ƫ�ȡ���ȵ���Ϣ'.format(num_col, cat_col))
    print('ע����ܻ����infֵ���ۼӹ���')
    features_ori = data.columns
    for col_1 in tqdm(cat_col):
        Train_gb = data.groupby(col_1)
        all_info = {}
        last = ''
        for kind, kind_data in Train_gb:
            info = {}
            for col_2 in num_col:
                # kind_data = kind_data[kind_data[col_2] > 0]
                info['%s_amount' % col_1] = len(kind_data)
                info['%s_%s_max' % (col_1, col_2)] = kind_data.loc[:, col_2].max()
                info['%s_%s_median' % (col_1, col_2)] = kind_data.loc[:, col_2].median()
                info['%s_%s_min' % (col_1, col_2)] = kind_data.loc[:, col_2].min()
                info['%s_%s_sum' % (col_1, col_2)] = kind_data.loc[:, col_2].sum()
                info['%s_%s_std' % (col_1, col_2)] = kind_data.loc[:, col_2].std()
                info['%s_%s_average' % (col_1, col_2)] = round(kind_data.loc[:, col_2].sum() / (len(kind_data) + 1), 2)
                info['%s_%s_skew' % (col_1, col_2)] = kind_data.loc[:, col_2].skew()
                info['%s_%s_kurt' % (col_1, col_2)] = kind_data.loc[:, col_2].kurt()
                all_info[kind] = info
                last = col_2
        df_new = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": col_1})
        data = data.merge(df_new, how='left', on=col_1)
    if add_new:
        return data
    else:
        features_all = data.columns.tolist()
        features_new = [f for f in features_all if f not in features_ori]
        data = data.loc[:, features_new]
        return data


#��ֵ���������������������ͳ��
def cross_cat_num(df, num_col, cat_col, add_new=True):
    print('��������{}����������{}�����ͳ�����ֵ����Сֵ����ֵ����Ϣ'.format(num_col, cat_col))
    features_ori = df.columns
    for f1 in tqdm(cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in tqdm(num_col):
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max', '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_median'.format(f1, f2): 'median',
            })
            df = df.merge(feat, on=f1, how='left')

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new

# ���㹲�ִ�����n unique���ء�����ƫ��
def cross_qua_cat_num(df, comp_features, add_new=True):
    print('��ȡ�����ԣ�{}���ִ�����n unique���ء�����ƫ�õ���Ϣ'.format(comp_features))
    features_ori = df.columns
    for f_pair in tqdm(comp_features):
        # print('������{} + ������{}'.format(f_pair[0], f_pair[1]))
        ### ���ִ���
        df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)[f_pair[0]].transform('count')
        ### n unique����
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
        df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
            '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
            '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[1], how='left')
        ### ����ƫ��
        # df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
        # df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new
