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
用于训练集、测试集合并后，一起构建新特征
"""

######################################################时间特征######################################################
#日期提取
def date_get_ymdw(df, fea_col, add_new=True):
    print('提取时间特征：{}，提取年、月、日、周等信息'.format(fea_col))
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

# 构造时间特征
def date_features(df, f1, f2, add_new=True):
    """
    f1-f2的日期差，当前日期与f1的时间差，当前日期与f2的日期差
    :param df:
    :param f1:
    :param f2:
    :return:
    """
    print('提取时间特征：{}与{}的时间差'.format(f1, f2))
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



######################################################数值特征######################################################
#计算特征统计量
import math
def produce_by_single(data, num_cols):
    print('提取特征：{}平方、立方、对数等信息'.format(num_cols))
    for i in num_cols:
        data[i + '**2'] = data[i].map(lambda x: x**2)

    for i in num_cols:
        data[i + '**3'] = data[i].map(lambda x: x**3)

    # for i in num_cols:
    #     # data['log_' + i] = data[i].map(lambda x: math.log(x+1))
    #     data['log_' + i] = np.log1p(data[i])
    return data

def produce_by_double(data, num_cols):
    print('提取特征：{}相互交叉加减乘数后的信息'.format(num_cols))
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


#分桶操作
def cut_group(df, cols, num_bins=50, add_new=True):
    print('对特征：{}进行分桶（{}）操作'.format(cols, num_bins))
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



######################################################类别特征######################################################
### count编码
def count_coding(df, fea_col, add_new=True):
    print('对特征：{}进行count编码'.format(fea_col))
    features_ori = df.columns
    for f in fea_col:
        df[f + '_count'] = df[f].map(df[f].value_counts())

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new

# 获取onehot编码
def gen_onehot(df, col_name='F', deal_with_sp=False):
    """
    将df对特征col_name进行编码
    对比：pd.get_dummies(data, columns=col_names)
    不同点：gen_onehot可处理特殊的字符
    example:
            a1 = pd.Series(['我','@','c'])
            b1 = pd.Series([1,2,3])
            c1 = pd.DataFrame({'F1':a1,'F2':b1})

            print(c1)
            r = gen_onehot(c1,col_name='F1')
            print(r)
            t = pd.get_dummies(c1, columns=['F1','F2'])
            print(t)

            output:

                  F1  F2
                0  我   1
                1  @   2
                2  c   3
                   F1_NaN  F1_c  F1_我
                0       0     0     1
                1       1     0     0
                2       0     1     0
                   F1_@  F1_c  F1_我  F2_1  F2_2  F2_3
                0     0     0     1     1     0     0
                1     1     0     0     0     1     0
                2     0     1     0     0     0     1
    """

    if col_name not in df.columns:
        print('请确认特征{}在该数据集中'.format(col_name))
        return df
    else:
        data_df = df
        data_with_new_features = pd.get_dummies(data_df[col_name])

        if deal_with_sp:
            # 新特征重命名
            new_cols_names = []
            cols = 0
            for col in data_with_new_features.columns.values.tolist():
                str_sub = re.sub("[\#\@\$\&\*\(\)\\\~\`\<\>\.\-\_\!\%\[\]\,\。]", "", col)
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

# 用onehot编码取代原特征
def add_onehot_features(data_ori, cols_to_onehot, add_new=True):
    """
    将Onehot编码后的特征取代旧特征
    :param data_ori:
    :param cols_to_onehot:list
    :return:
    """
    print('对特征：{}进行onehot编码'.format(cols_to_onehot))
    features_ori = data_ori.columns
    data = data_ori.copy()
    for c in cols_to_onehot:
        print('\n对特征：{}进行onehot编码。。。'.format(c))
        data_with_new_features = gen_onehot(data, col_name=c)
        print('新增onehot编码特征：')
        print(data_with_new_features.columns.values.tolist())
        print('\n')
        # 用编码特征取代原特征
        data = data.drop(columns=[c], axis=1)
        data = pd.concat([data, data_with_new_features], axis=1)


    if add_new:
        return data
    else:
        features_new = [f for f in data.columns if f not in features_ori]
        df_new = data.loc[:, features_new]
        return df_new



######################################################交叉特征######################################################
def produce_by_statistics(data, num_col, cat_col, add_new=True):
    """
    分组后计算特征统计量作为新特征
    :param data: dataFrame格式
    :param col_1: 用于分组的特征
    :param col_2: 用于统计的特征
    :return: brand_fe：新特征；data：合并新特征后的数据集
    """
    print('对特征：{}，按照特征：{}分组后'
          '统计总数、最大值、最小值、中值、总和、均值、方差、偏度、峰度等信息'.format(num_col, cat_col))
    print('注意可能会产生inf值（累加过大）')
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


#数值特征和类别特征交叉特征统计
def cross_cat_num(df, num_col, cat_col, add_new=True):
    print('对特征：{}按照特征：{}分组后统计最大值、最小值、中值等信息'.format(num_col, cat_col))
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

# 计算共现次数、n unique、熵、比例偏好
def cross_qua_cat_num(df, comp_features, add_new=True):
    print('提取特征对：{}共现次数、n unique、熵、比例偏好等信息'.format(comp_features))
    features_ori = df.columns
    for f_pair in tqdm(comp_features):
        # print('特征：{} + 特征：{}'.format(f_pair[0], f_pair[1]))
        ### 共现次数
        df['_'.join(f_pair) + '_count'] = df.groupby(f_pair)[f_pair[0]].transform('count')
        ### n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
        df = df.merge(df.groupby(f_pair[1], as_index=False)[f_pair[0]].agg({
            '{}_{}_nunique'.format(f_pair[1], f_pair[0]): 'nunique',
            '{}_{}_ent'.format(f_pair[1], f_pair[0]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[1], how='left')
        ### 比例偏好
        # df['{}_in_{}_prop'.format(f_pair[0], f_pair[1])] = df['_'.join(f_pair) + '_count'] / df[f_pair[1] + '_count']
        # df['{}_in_{}_prop'.format(f_pair[1], f_pair[0])] = df['_'.join(f_pair) + '_count'] / df[f_pair[0] + '_count']

    if add_new:
        return df
    else:
        features_new = [f for f in df.columns if f not in features_ori]
        df_new = df.loc[:, features_new]
        return df_new
