# coding=gbk

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Analysis_all_features(object):
    """
    ���������ķ���
    ѵ�����У�cols��label��features�ļ���
    ���Լ��У�cols��features
    """
    def __init__(self, data, cols, label=None, train=True):
        self.data = data
        self.cols = cols.copy()
        self.features = cols.copy()
        self.istrain = train
        if self.istrain:
            # ѵ�������ݣ���δָ��label��Ĭ�����һ����Ϊlabel
            if not label:
                self.label = cols[-1]
                print("��ȷ���Ƿ��� '{}'��Ϊlabel".format(self.label))
                self.features.pop()
            else:
                self.label = label
                # ���cols���Ƿ��Ѿ�����label
                if label not in self.cols:
                    self.cols.append(label)
                else:
                    self.features.remove(label)
        else:
            # ���Լ����ݣ�����Ҫָ��label
            if not label:
                print("���Լ��в���Ҫlabel")
                self.label = None

    def show_shape(self):
        """
        ͳ����������������
        :return:
        """
        print('\n*******************************ͳ����������������*******************************')
        row_num,ncol = self.data.shape
        print('���ݼ���������{}��������{}\n\n'.format(row_num,ncol))

    def percentage_miss(self):
        """
        ����ֵ���ࡢȱʧֵռ�ȡ��������ֵռ�ȡ��������ͷ���
        :return:
        """
        print('\n************ͳ������ֵ���ࡢȱʧֵռ�ȡ��������ֵռ�ȡ��������ͷ���************')
        # DataFrame��ӡ�������
        # ��ʾ������
        pd.set_option('display.max_columns', None)
        # ��ʾ������
        pd.set_option('display.max_rows', None)

        train_stats = []
        for col in self.features:
            train_stats.append((col, self.data[col].nunique(), self.data[col].isnull().sum()*100/self.data.shape[0],
                                self.data[col].value_counts(normalize=True, dropna=False).values[0]*100,
                                self.data[col].dtype))


        stats_df = pd.DataFrame(train_stats, columns=['Feature', 'Unique_values', 'P of NaN',
                                                      'P of biggest category', 'type'])
        stats_df.sort_values('P of NaN', ascending=False)
        # print('\n\n����ֵ���ࡢȱʧֵռ�ȡ��������ֵռ�ȡ��������ͣ�')
        print(stats_df)
        # DataFrame��ӡ�������
        # ��ʾ������
        pd.set_option('display.max_columns', 20)
        # ��ʾ������
        pd.set_option('display.max_rows', 60)

    def show_corr_label_with_features(self, show_picture=False):
        """
        feature��label��Ƥ��ѷ�����
        :param show_picture:
        :return:
        """
        print('\n************************����feature��label��Ƥ��ѷ�����************************')
        if not self.istrain:
            print('���ڲ��Լ�����Ҫ����feature��label��Ƥ��ѷ�����')
        else:
            label_numeric = self.data[self.cols]
            correlation = label_numeric.corr()
            # print('\n\nfeature��label��Ƥ��ѷ�����:')
            print(correlation[self.label].sort_values(ascending=False), '\n')
            if show_picture:
                f, ax = plt.subplots(figsize=(7, 7))
                plt.title('Correlation of Numeric Features with %s' % self.label, y=1, size=16)
                sns.heatmap(correlation, square=True, vmax=0.8)
                plt.show()




class Analysis_categorical_features(object):
    """
    ��������ķ���
    ѵ�����У�cols��label��features�ļ���
    ���Լ��У�cols��features
    """
    def __init__(self, data, cols, label=None, train=True):
        self.data = data
        self.cols = cols.copy()
        self.features = cols.copy()
        self.istrain = train
        if self.istrain:
            # ѵ�������ݣ���δָ��label��Ĭ�����һ����Ϊlabel
            if not label:
                self.label = cols[-1]
                print("��ȷ���Ƿ��� '{}'��Ϊlabel".format(self.label))
                self.features.pop()
            else:
                self.label = label
                # ���cols���Ƿ��Ѿ�����label
                if label not in self.cols:
                    self.cols.append(label)
                else:
                    self.features.remove(label)
        else:
            # ���Լ����ݣ�����Ҫָ��label
            if not label:
                print("���Լ��в���Ҫlabel")
                self.label = None

    def show_features_distribution(self):
        """
        ÿ����������ͬ����ֵ������ͳ��
        :return:
        """
        print('\n********************��ͬ����������ֵͳ��********************')
        for cat_fea in self.features:
            series_value_counts = self.data[cat_fea].value_counts()

            labels = series_value_counts.index.tolist()
            values = series_value_counts.values

            fig = plt.figure()
            plt.pie(values, labels=labels, autopct='%1.2f%%')  # ����ͼ�����ݣ����ݶ�Ӧ�ı�ǩ���ٷ���������λС���㣩
            plt.title("Pie chart of "+ cat_fea)
            # plt.show()


            print(cat_fea + "�������ֲ����£�")
            print("{}�����и�{}��ͬ��ֵ".format(cat_fea, self.data[cat_fea].nunique()))
            print(series_value_counts)
            print('\n\n')
        plt.show()

    def view_of_violinplot(self):
        """
        ����С����ͼ
        :return:
        """
        print('\n************************����С����ͼ************************')
        for catg in self.features:
            sns.violinplot(x=catg, y=self.label, data=self.data)
            plt.title(catg+' violinplot')
            plt.show()

    def view_of_count_plot(self):
        """
        ���������ÿ�����Ƶ��
        :return:
        """
        print('\n************************�������Ƶ��************************')
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
        �������������������ת��Ϊcategory��ͬʱ����NaN����
        :return:
        """
        print('\n********************ת�����������������********************')
        for c in self.features:
            # print('************feature:'+c)
            self.data[c] = self.data[c].astype('category')
            if self.data[c].isnull().any():
                self.data[c] = self.data[c].cat.add_categories(['MISSING'])
                self.data[c] = self.data[c].fillna('MISSING')



class Analysis_numeric_features(object):
    """
    ��ֵ�����ķ���
    ѵ�����У�cols��label��features�ļ���
    ���Լ��У�cols��features
    """
    def __init__(self, data, cols, label=None, train=True):
        self.data = data
        self.cols = cols.copy()
        self.features = cols.copy()
        self.istrain = train
        if self.istrain:
            # ѵ�������ݣ���δָ��label��Ĭ�����һ����Ϊlabel
            if not label:
                self.label = cols[-1]
                print("��ȷ���Ƿ��� '{}'��Ϊlabel".format(self.label))
                self.features.pop()
            else:
                self.label = label
                # ���cols���Ƿ��Ѿ�����label
                if label not in self.cols:
                    self.cols.append(label)
                else:
                    self.features.remove(label)
        else:
            # ���Լ����ݣ�����Ҫָ��label
            if not label:
                print("���Լ��в���Ҫlabel")
                self.label = None

    def show_corr_label_with_features(self, show_picture=False):
        """
        features��label�ģ�Ƥ��ѷ���ԣ���ض�
        :param show_picture:
        :return:
        """
        print('\n********������ֵ������label�ģ�Ƥ��ѷ���ԣ���ض�********')
        if not self.istrain:
            print('���ڲ��Լ�����Ҫ����feature��label��Ƥ��ѷ�����')
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
        ƫ�ȣ�Skewness��--�������ݷֲ���̬��ͳ����������������ĳ����ȡֵ�ֲ��ĶԳ��ԣ�����˵�������ݵĲ��ԳƳ̶ȡ�
        ��1��Skewness = 0 ���ֲ���̬����̬�ֲ�ƫ����ͬ��
        ��2��Skewness > 0 ����ƫ����ֵ�ϴ�Ϊ��ƫ����ƫ����β�������ұߣ������Ҷ��н϶�ļ���ֵ��
        ��3��Skewness < 0 ����ƫ����ֵ�ϴ�Ϊ��ƫ����ƫ����β��������ߣ���������н϶�ļ���ֵ��
        ��4����ֵ�ľ���ֵԽ�󣬱������ݷֲ�Խ���Գƣ�ƫб�̶ȴ�

        ��ȣ�Kurtosis��--����ĳ��������ȡֵ�ֲ���̬�����̶ȵ�ͳ����������˵�������ݷֲ����ļ���̶ȡ�
        ��1��Kurtosis=0 ����̬�ֲ��Ķ����̶���ͬ��
        ��2��Kurtosis>0 ����̬�ֲ��ĸ߷���Ӷ��͡����ⶥ��
        ��3��Kurtosis<0 ����̬�ֲ��ĸ߷�����ƽ̨����ƽ����

        :param show_picture:
        :return:
        """
        print('\n******************������ֵ����ƫ�Ⱥͷ��******************')
        for col in self.features:
            print('{:15}'.format(col),
                  'Skewness: {:05.2f}'.format(self.data[col].skew()),
                  '   ',
                  'Kurtosis: {:06.2f}'.format(self.data[col].kurt())
                  )


    def view_of_box(self):
        """
        ��������ͼ
        :return:
        """
        print('\n************************��������ͼ************************')
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
        ������״ͼ���������ⲻ����ʹ��
        :return:
        """
        print('\n************************������״ͼ************************')
        def show_bar_plot(x, y, **kwargs):
            # classLabel��ʾlabel�Ƿ�Ϊ��������
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
        �ֲ��������ʷֲ����ߣ�
        :return:
        """
        print('\n***������ֵ������label�ķֲ��������ʣ��ֲ�����&�������***')
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
        �����໥֮��Ĺ�ϵ���ӻ�
        :return:
        """
        print('\n***************��ֵ�����໥֮��Ĺ�ϵ���ӻ�***************')
        sns.set()
        sns.pairplot(self.data[self.features], height=2, kind='scatter', diag_kind='kde')
        plt.show()
