# coding=gbk

import pandas as pd
from analysis.untils import dataAnalysis


def analysis_main(data, columns, labels=None, categorical_features=None, num_features=None):
    ## 所有特征的分析
    print('##################################################################################')
    print('###################################处理所有特征###################################')
    print('##################################################################################')
    analysis_all_features = dataAnalysis.Analysis_all_features(data, columns, labels)
    analysis_all_features.show_shape()
    analysis_all_features.percentage_miss()
    analysis_all_features.show_corr_label_with_features(show_picture=False)


    ## 类别特征的分析
    print('\n\n\n')
    print('##################################################################################')
    print('###################################处理类别特征###################################')
    print('##################################################################################')
    if not categorical_features:
        print('\n请指定需要分析的类别特征！！！')
    else:
        analysis_categorical_features = dataAnalysis.Analysis_categorical_features(data, categorical_features, labels)
        analysis_categorical_features.show_features_distribution()
        analysis_categorical_features.view_of_violinplot()
        analysis_categorical_features.view_of_count_plot()
        analysis_categorical_features.change_datatype()


    ## 数值特征的分析
    print('\n\n\n')
    print('##################################################################################')
    print('###################################处理数值特征###################################')
    print('##################################################################################')
    if not num_features:
        print('\n请指定需要分析的数值特征！！！')
    else:
        analysis_numeric_features = dataAnalysis.Analysis_numeric_features(data, num_features, labels)
        analysis_numeric_features.show_corr_label_with_features(show_picture=True)
        analysis_numeric_features.show_Skew_and_kurt()
        analysis_numeric_features.view_of_box()
        # classLabel为False时不建议使用
        # analysis_numeric_features.view_of_bar_plot(classLabel=False)
        analysis_numeric_features.view_of_distplot()
        analysis_numeric_features.view_of_pairplot()


    print('=======================> 数据分析过程结束！！！')

