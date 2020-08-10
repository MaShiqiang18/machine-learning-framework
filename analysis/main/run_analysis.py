# coding=gbk

import pandas as pd
from analysis.untils import dataAnalysis


def analysis_main(data, columns, labels=None, categorical_features=None, num_features=None):
    ## ���������ķ���
    print('##################################################################################')
    print('###################################������������###################################')
    print('##################################################################################')
    analysis_all_features = dataAnalysis.Analysis_all_features(data, columns, labels)
    analysis_all_features.show_shape()
    analysis_all_features.percentage_miss()
    analysis_all_features.show_corr_label_with_features(show_picture=False)


    ## ��������ķ���
    print('\n\n\n')
    print('##################################################################################')
    print('###################################�����������###################################')
    print('##################################################################################')
    if not categorical_features:
        print('\n��ָ����Ҫ�������������������')
    else:
        analysis_categorical_features = dataAnalysis.Analysis_categorical_features(data, categorical_features, labels)
        analysis_categorical_features.show_features_distribution()
        analysis_categorical_features.view_of_violinplot()
        analysis_categorical_features.view_of_count_plot()
        analysis_categorical_features.change_datatype()


    ## ��ֵ�����ķ���
    print('\n\n\n')
    print('##################################################################################')
    print('###################################������ֵ����###################################')
    print('##################################################################################')
    if not num_features:
        print('\n��ָ����Ҫ��������ֵ����������')
    else:
        analysis_numeric_features = dataAnalysis.Analysis_numeric_features(data, num_features, labels)
        analysis_numeric_features.show_corr_label_with_features(show_picture=True)
        analysis_numeric_features.show_Skew_and_kurt()
        analysis_numeric_features.view_of_box()
        # classLabelΪFalseʱ������ʹ��
        # analysis_numeric_features.view_of_bar_plot(classLabel=False)
        analysis_numeric_features.view_of_distplot()
        analysis_numeric_features.view_of_pairplot()


    print('=======================> ���ݷ������̽���������')

