# coding=gbk

from preProcess.untils import dataPreprocess

"""
���ݷ�������������ǰ����������Ԥ����
��Ҫ���ڣ��쳣ֵ������ֵ������ֵ����ת����ʱ��������
"""


def preprocess_main(data_ori, categorical_features=None, num_features=None, time_features=None, fillnan=True):
    """
    ����Ԥ������Ҫ�����쳣ֵ����(ʶ���쳣ֵ����ɾ��)����ֵ������������ת��
    :param data_ori: ԭʼ����
    :param columns: ����������
    :param labels: ��ǩ��
    :param categorical_features: �������
    :param num_features: ��ֵ����
    :param time_features: ʱ������
    :param fillnan: �Ƿ����ȱʧֵ
    :return:
    """
    data = data_ori.copy()
    columns = data.columns.tolist()

    print('\n\n******************������ֵ�������쳣ֵ******************')
    abnormal_index = []
    for f in num_features:
        print('������'+f)
        index_f = dataPreprocess.outliers_proc(data, f, scale=3, comp=False, show_view=False)
        abnormal_index.extend(index_f.tolist())
    abnormal_index_set = set(abnormal_index)
    print('\n====>ʶ����ܹ�{}������'.format(len(abnormal_index_set)))


    num_null = data.isnull().sum().sum()
    if (num_null > 0) & fillnan:
        print('\n\n************************�����ֵ************************')
        ### �����ֵ��-���Ĵ����յ�ָ������
        # sp_feature = ''
        # concat_data[sp_feature] = concat_data[sp_feature].replace('-', 0).astype('float16')
        data = dataPreprocess.replace_var(data, cols_name=columns)
        num_null = data.isnull().sum()
        if num_null.sum() != 0:
            for f in columns:
                if num_null[f] != 0:
                    print('����{}�п�ֵ��{}'.format(f, num_null[f]))
        else:
            print('��ֵ�����ϣ�')

    print('\n\n**********************ת����������**********************')
    data = dataPreprocess.reduce_mem_usage(data, time_features)

    return data, abnormal_index_set
