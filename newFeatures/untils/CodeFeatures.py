# coding=gbk

from newFeatures.untils.MeanEncoder import MeanEncoder
from sklearn.model_selection import KFold
from tqdm import tqdm


"""
����ѵ���������Լ��ֱ𹹽���������������������
1. ѵ�����ȱ���
2. ���Լ�����ѵ��������
"""

def meanEnocoding(X_data,Y_data,X_test, Feature_list):
    """
    Feature_list ����ֵ�������漸�㣺1. ���ظ���2. ������ͬ��ֵ�����ֳ�����һ������������100������.
    """
    # ������Ҫƽ�������������
    MeanEnocodeFeature = Feature_list
    # ����ƽ�����������
    ME = MeanEncoder(MeanEnocodeFeature, target_type='regression')
    # ��ѵ�����ݼ���X��y�������
    X_train = ME.fit_transform(X_data, Y_data)
    # ��ѵ�����ݼ���X��y�������
    #x_train_fav = ME.fit_transform(x_train,y_train_fav)
    # �Բ��Լ����б���
    X_test = ME.transform(X_test)

    return X_train, X_test




### target encodingĿ����룬�ع鳡�������˵��Ŀ������ѡ����࣬������������ֵ���룬����������׼����롢��λ�������
def targetEncoding(X_data,Y_data,X_test, col, group_by_features, enc_stats=None):
    """
    ���Ϊ10�ۺ�ÿһ�۰�group_by_features�е�һ���������飬
    ȡ�������ÿһ���col�������С����ֵ����enc_stats������Ϊ������
    :param X_data:
    :param Y_data:
    :param X_test:
    :param col: ��ֵ����
    :param group_by_features: �������
    :param enc_stats: stats_default_dict�е�keys��ѡ
    :return:
    """
    index_ori = X_data.index
    X_data.reset_index(drop=True, inplace=True)

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
    ### ����ѡ�������ֱ���
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

    X_data.index = index_ori
    return X_data, X_test




