# coding=gbk

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Analysis_all_features(object):
    """
    所有特征的分析
    训练集中：cols是label和features的集合
    测试集中：cols即features
    """
    def __init__(self, data, cols, label=None, train=True):
        self.data = data
        self.cols = cols.copy()
        self.features = cols.copy()
        self.istrain = train
        if self.istrain:
            # 训练集数据，若未指定label，默认最后一列作为label
            if not label:
                self.label = cols[-1]
                print("请确认是否以 '{}'作为label".format(self.label))
                self.features.pop()
            else:
                self.label = label
                # 检查cols中是否已经包括label
                if label not in self.cols:
                    self.cols.append(label)
                else:
                    self.features.remove(label)
        else:
            # 测试集数据，不需要指定label
            if not label:
                print("测试集中不需要label")
                self.label = None

    def show_shape(self):
        """
        统计样本数，特征数
        :return:
        """
        print('\n*******************************统计样本数，特征数*******************************')
        row_num,ncol = self.data.shape
        print('数据集有样本：{}，特征：{}\n\n'.format(row_num,ncol))

    def percentage_miss(self):
        """
        特征值种类、缺失值占比、最大特征值占比、特征类型分析
        :return:
        """
        print('\n************统计特征值种类、缺失值占比、最大特征值占比、特征类型分析************')
        # DataFrame打印输出设置
        # 显示所有列
        pd.set_option('display.max_columns', None)
        # 显示所有行
        pd.set_option('display.max_rows', None)

        train_stats = []
        for col in self.features:
            train_stats.append((col, self.data[col].nunique(), self.data[col].isnull().sum()*100/self.data.shape[0],
                                self.data[col].value_counts(normalize=True, dropna=False).values[0]*100,
                                self.data[col].dtype))


        stats_df = pd.DataFrame(train_stats, columns=['Feature', 'Unique_values', 'P of NaN',
                                                      'P of biggest category', 'type'])
        stats_df.sort_values('P of NaN', ascending=False)
        # print('\n\n特征值种类、缺失值占比、最大特征值占比、特征类型：')
        print(stats_df)
        # DataFrame打印输出设置
        # 显示所有列
        pd.set_option('display.max_columns', 20)
        # 显示所有行
        pd.set_option('display.max_rows', 60)

    def show_corr_label_with_features(self, show_picture=False):
        """
        feature与label的皮尔逊相关性
        :param show_picture:
        :return:
        """
        print('\n************************计算feature与label的皮尔逊相关性************************')
        if not self.istrain:
            print('对于测试集不需要分析feature与label的皮尔逊相关性')
        else:
            label_numeric = self.data[self.cols]
            correlation = label_numeric.corr()
            # print('\n\nfeature与label的皮尔逊相关性:')
            print(correlation[self.label].sort_values(ascending=False), '\n')
            if show_picture:
                f, ax = plt.subplots(figsize=(7, 7))
                plt.title('Correlation of Numeric Features with %s' % self.label, y=1, size=16)
                sns.heatmap(correlation, square=True, vmax=0.8)
                plt.show()




class Analysis_categorical_features(object):
    """
    类别特征的分析
    训练集中：cols是label和features的集合
    测试集中：cols即features
    """
    def __init__(self, data, cols, label=None, train=True):
        self.data = data
        self.cols = cols.copy()
        self.features = cols.copy()
        self.istrain = train
        if self.istrain:
            # 训练集数据，若未指定label，默认最后一列作为label
            if not label:
                self.label = cols[-1]
                print("请确认是否以 '{}'作为label".format(self.label))
                self.features.pop()
            else:
                self.label = label
                # 检查cols中是否已经包括label
                if label not in self.cols:
                    self.cols.append(label)
                else:
                    self.features.remove(label)
        else:
            # 测试集数据，不需要指定label
            if not label:
                print("测试集中不需要label")
                self.label = None

    def show_features_distribution(self):
        """
        每个特征，不同特征值的数量统计
        :return:
        """
        print('\n********************不同特征，特征值统计********************')
        for cat_fea in self.features:
            series_value_counts = self.data[cat_fea].value_counts()

            labels = series_value_counts.index.tolist()
            values = series_value_counts.values

            fig = plt.figure()
            plt.pie(values, labels=labels, autopct='%1.2f%%')  # 画饼图（数据，数据对应的标签，百分数保留两位小数点）
            plt.title("Pie chart of "+ cat_fea)
            # plt.show()


            print(cat_fea + "的特征分布如下：")
            print("{}特征有个{}不同的值".format(cat_fea, self.data[cat_fea].nunique()))
            print(series_value_counts)
            print('\n\n')
        plt.show()

    def view_of_violinplot(self):
        """
        绘制小提琴图
        :return:
        """
        print('\n************************绘制小提琴图************************')
        for catg in self.features:
            sns.violinplot(x=catg, y=self.label, data=self.data)
            plt.title(catg+' violinplot')
            plt.show()

    def view_of_count_plot(self):
        """
        类别特征的每个类别频数
        :return:
        """
        print('\n************************绘制类别频数************************')
        def show_count_plot(x,**kwargs):
            sns.countplot(x=x)
            x = plt.xticks(rotation=90)
            plt.title(kwargs['f']+' count')
            plt.show()

        for c in self.features:
            f = pd.melt(self.data, value_vars=c)
            g = sns.FacetGrid(f, col="variable", col_wrap=1, sharex=False, sharey=False, height=5)
            g = g.map(show_count_plot, "value", f=c)

    def change_datatype(self):
        """
        将类别特征的数据类型转化为category，同时处理NaN数据
        :return:
        """
        print('\n********************转化类别特征数据类型********************')
        for c in self.features:
            # print('************feature:'+c)
            self.data[c] = self.data[c].astype('category')
            if self.data[c].isnull().any():
                self.data[c] = self.data[c].cat.add_categories(['MISSING'])
                self.data[c] = self.data[c].fillna('MISSING')



class Analysis_numeric_features(object):
    """
    数值特征的分析
    训练集中：cols是label和features的集合
    测试集中：cols即features
    """
    def __init__(self, data, cols, label=None, train=True):
        self.data = data
        self.cols = cols.copy()
        self.features = cols.copy()
        self.istrain = train
        if self.istrain:
            # 训练集数据，若未指定label，默认最后一列作为label
            if not label:
                self.label = cols[-1]
                print("请确认是否以 '{}'作为label".format(self.label))
                self.features.pop()
            else:
                self.label = label
                # 检查cols中是否已经包括label
                if label not in self.cols:
                    self.cols.append(label)
                else:
                    self.features.remove(label)
        else:
            # 测试集数据，不需要指定label
            if not label:
                print("测试集中不需要label")
                self.label = None

    def show_corr_label_with_features(self, show_picture=False):
        """
        features与label的（皮尔逊线性）相关度
        :param show_picture:
        :return:
        """
        print('\n********计算数值特征与label的（皮尔逊线性）相关度********')
        if not self.istrain:
            print('对于测试集不需要分析feature与label的皮尔逊相关性')
        else:
            label_numeric = self.data[self.cols]
            correlation = label_numeric.corr()
            print(correlation[self.label].sort_values(ascending=False), '\n')
            if show_picture:
                f, ax = plt.subplots(figsize=(7, 7))
                plt.title('Correlation of Numeric Features with %s' % self.label, y=1, size=16)
                sns.heatmap(correlation, square=True, vmax=0.8)
                plt.show()


    def show_Skew_and_kurt(self):
        """
        偏度（Skewness）--描述数据分布形态的统计量，其描述的是某总体取值分布的对称性，简单来说就是数据的不对称程度。
        （1）Skewness = 0 ，分布形态与正态分布偏度相同。
        （2）Skewness > 0 ，正偏差数值较大，为正偏或右偏。长尾巴拖在右边，数据右端有较多的极端值。
        （3）Skewness < 0 ，负偏差数值较大，为负偏或左偏。长尾巴拖在左边，数据左端有较多的极端值。
        （4）数值的绝对值越大，表明数据分布越不对称，偏斜程度大。

        峰度（Kurtosis）--描述某变量所有取值分布形态陡缓程度的统计量，简单来说就是数据分布顶的尖锐程度。
        （1）Kurtosis=0 与正态分布的陡缓程度相同。
        （2）Kurtosis>0 比正态分布的高峰更加陡峭——尖顶峰
        （3）Kurtosis<0 比正态分布的高峰来得平台——平顶峰

        :param show_picture:
        :return:
        """
        print('\n******************计算数值特征偏度和峰度******************')
        for col in self.features:
            print('{:15}'.format(col),
                  'Skewness: {:05.2f}'.format(self.data[col].skew()),
                  '   ',
                  'Kurtosis: {:06.2f}'.format(self.data[col].kurt())
                  )


    def view_of_box(self):
        """
        绘制箱型图
        :return:
        """
        print('\n************************绘制箱型图************************')
        def show_boxplot(x, y, **kwargs):
            if kwargs['classLabel']:
                sns.boxplot(x=x, y=y)
                x = plt.xticks(rotation=90)
                plt.title(kwargs['f'] + ' boxplot')
                plt.show()
            else:
                sns.boxplot(x=x)
                x = plt.xticks(rotation=90)
                plt.title(kwargs['f'] + ' boxplot')
                plt.show()

        for c in self.features:
            f = pd.melt(self.data, id_vars=[self.label], value_vars=c)
            g = sns.FacetGrid(f, col="variable", col_wrap=1, sharex=False, sharey=False, height=5)
            g = g.map(show_boxplot, "value", self.label, f=c, classLabel=False)


    def view_of_bar_plot(self, classLabel):
        """
        绘制柱状图，分类问题不建议使用
        :return:
        """
        print('\n************************绘制柱状图************************')
        def show_bar_plot(x, y, **kwargs):
            # classLabel表示label是否为分类问题
            if kwargs['classLabel']:
                sns.barplot(x=x, y=y)
                x = plt.xticks(rotation=90)
                plt.title(kwargs['c']+' bar')
                plt.show()
            else:
                sns.barplot(x=x)
                x = plt.xticks(rotation=90)
                plt.title(kwargs['c']+' bar')
                plt.show()

        for c in self.features:
            f = pd.melt(self.data, id_vars=[self.label], value_vars=c)
            g = sns.FacetGrid(f, col="variable", col_wrap=1, sharex=False, sharey=False, height=5)
            g = g.map(show_bar_plot, "value", self.label, c=c, classLabel=classLabel)


    def view_of_distplot(self):
        """
        分布（含概率分布曲线）
        :return:
        """
        print('\n***绘制数值特征和label的分布（含概率）分布曲线&线性拟合***')
        f = pd.melt(self.data, value_vars=self.cols)
        g = sns.FacetGrid(f, col="variable", col_wrap=2, sharex=False, sharey=False)
        g = g.map(sns.distplot, "value")
        plt.show()

        sns.pairplot(self.data[self.cols], height=2, kind='scatter', diag_kind='kde')
        plt.show()

        fig, ax = plt.subplots(nrows=len(self.cols)//2, ncols=2,figsize=(24, 20))
        # ['v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
        for c in range(1,len(self.features)+1):
            scatter_plot = pd.concat([self.data[self.label], self.data[self.features]], axis=1)
            if c % 2 != 0:
                p = (c + 1) // 2
                ax_c = ax[p - 1][0]
            else:
                p = c // 2
                ax_c = ax[p - 1][1]
            f = self.features[c-1]
            sns.regplot(x=f, y=self.label, data=scatter_plot, scatter=True, fit_reg=True, ax=ax_c)

        plt.show()

    def view_of_pairplot(self):
        """
        特征相互之间的关系可视化
        :return:
        """
        print('\n***************数值特征相互之间的关系可视化***************')
        sns.set()
        sns.pairplot(self.data[self.features], height=2, kind='scatter', diag_kind='kde')
        plt.show()
