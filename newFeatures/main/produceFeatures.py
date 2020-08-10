# coding=gbk

from newFeatures.untils import NewFeatures
import numpy as np

def featureProcess_main(data_ori, columns, labels=None, categorical_features=None, num_features=None,
                        count_coding_features=None, tocut_features=None, time_features=None, onehot_features=None,
                        comp_features=None, cross_cat=None, cross_num=None):
    """
    特征工程，按照原有的数值特征、类别特征，分别构建新特征
    :param data_ori: 初始数据
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

    # 时间特征处理
    if time_features:
        print('\n\n******************从时间特征中提取年、月、日、周作为新特征******************')
        data = NewFeatures.date_get_ymdw(data, time_features)
        if len(time_features) == 2:
            print('请注意是哪两个时期相减！！！')
            print('请确认是否是用特征：{} - {}'.format(time_features[0], time_features[1]))
            data = NewFeatures.date_features(data, time_features[0], time_features[1])


    ## 数值特征处理
    # if num_features:
    #     print('\n\n*******************************将指定特征统计*******************************')
    #     data = NewFeatures.produce_by_single(data, num_features)
    #     data = NewFeatures.produce_by_double(data, num_features)


    if tocut_features:
        print('\n\n*******************************将指定特征分桶*******************************')
        data = NewFeatures.cut_group(data, tocut_features)


    ## 类别特征处理
    if onehot_features:
        print('\n\n****************************将指定特征onehot编码****************************')
        data = NewFeatures.add_onehot_features(data, onehot_features)

    if count_coding_features:
        print('\n\n*****************************将指定特征count编码*****************************')
        data = NewFeatures.count_coding(data, count_coding_features)


    ## 数值特征集合类别特征处理
    if cross_cat:
        if cross_num:
            print('\n\n******************************定义交叉特征统计******************************')
            data = NewFeatures.produce_by_statistics(data, cross_num, cross_cat)

            for f in data.columns:
                num = np.isinf(data[f].values).sum()
                if num > 0:
                    print('特征：{}中有Inf，请及时处理！'.format(f))

    if (len(comp_features)>0):
        print('\n\n*********************************组合特征统计*********************************')
        data = NewFeatures.cross_qua_cat_num(data, comp_features)

    return data

