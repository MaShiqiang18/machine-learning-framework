# coding=gbk

from newFeatures.untils import NewFeatures
import numpy as np

def featureProcess_main(data_ori, columns, labels=None, categorical_features=None, num_features=None,
                        count_coding_features=None, tocut_features=None, time_features=None, onehot_features=None,
                        comp_features=None, cross_cat=None, cross_num=None):
    """
    �������̣�����ԭ�е���ֵ����������������ֱ𹹽�������
    :param data_ori: ��ʼ����
    :param columns:
    :param labels:
    :param categorical_features:
    :param num_features:
    :param count_coding_features:
    :param tocut_features:
    :param time_features:
    :param onehot_features:
    :param comp_features:
    :return:
    """
    data = data_ori.copy()

    # ʱ����������
    if time_features:
        print('\n\n******************��ʱ����������ȡ�ꡢ�¡��ա�����Ϊ������******************')
        data = NewFeatures.date_get_ymdw(data, time_features)
        if len(time_features) == 2:
            print('��ע����������ʱ�����������')
            print('��ȷ���Ƿ�����������{} - {}'.format(time_features[0], time_features[1]))
            data = NewFeatures.date_features(data, time_features[0], time_features[1])


    ## ��ֵ��������
    # if num_features:
    #     print('\n\n*******************************��ָ������ͳ��*******************************')
    #     data = NewFeatures.produce_by_single(data, num_features)
    #     data = NewFeatures.produce_by_double(data, num_features)


    if tocut_features:
        print('\n\n*******************************��ָ��������Ͱ*******************************')
        data = NewFeatures.cut_group(data, tocut_features)


    ## �����������
    if onehot_features:
        print('\n\n****************************��ָ������onehot����****************************')
        data = NewFeatures.add_onehot_features(data, onehot_features)

    if count_coding_features:
        print('\n\n*****************************��ָ������count����*****************************')
        data = NewFeatures.count_coding(data, count_coding_features)


    ## ��ֵ�������������������
    if cross_cat:
        if cross_num:
            print('\n\n******************************���彻������ͳ��******************************')
            data = NewFeatures.produce_by_statistics(data, cross_num, cross_cat)

            for f in data.columns:
                num = np.isinf(data[f].values).sum()
                if num > 0:
                    print('������{}����Inf���뼰ʱ����'.format(f))

    if (len(comp_features)>0):
        print('\n\n*********************************�������ͳ��*********************************')
        data = NewFeatures.cross_qua_cat_num(data, comp_features)

    return data

