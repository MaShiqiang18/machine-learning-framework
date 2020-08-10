# coding=gbk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
# 基模型
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
# 包裹式选择法
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE, RFECV
# 过滤式选择法
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile
# 嵌入式选择法
from sklearn.feature_selection import SelectFromModel
# 特征乱序
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from IPython.display import display

# 评价函数
from sklearn.metrics import mean_absolute_error,  make_scorer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy.stats import pearsonr
from minepy import MINE

import warnings
warnings.simplefilter("ignore")


class Select_features(object):
    """
    使用Filter、Wrapper、Embedded以及通过将特征值乱序等模式，选出最好的K个特征
    """
    def __init__(self, data, continuous_feature_names, label, score, k=5, c='R', showFig=True):
        self.data = data
        self.continuous_feature_names = continuous_feature_names
        self.label = label
        self.score = score
        self.K = k
        self.C = c
        self.showFig = showFig
        # 备选特征
        self.train_X = data[continuous_feature_names]
        self.train_y = data[label]
        self.numNull = self.train_X.isnull().sum().sum()
        self.numInf = np.isinf(self.train_X.values).sum()


    def dict_features_score(self,scores):
        scores = list(map(lambda x: round(x, 4), scores))
        print(sorted(dict(zip(self.continuous_feature_names, scores)).items(), key=lambda x: x[1], reverse=True))

    """
    【Filter model(过滤器)】-->过滤式选择[单变量分析]
    单变量特征:
        基于单一变量和目标y之间的关系，通过计算某个能够度量特征重要性的指标，然后选出重要性Top的K个特征。
    缺点: 忽略了特征组合的情况
    """
    # 方差选择法(移除低方差特征)
    def select_by_var(self, threshold=0.):
        # 移除那些方差低于某个阈值，即特征值变动幅度小于某个范围的特征，这一部分特征的区分度较差，进行移除。
        # 返回值为特征选择后的数据
        # 参数threshold为方差的阈值
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.train_X)
        data = selector.transform(self.train_X)

        featureVar = selector.variances_
        # print(featureVar)
        self.dict_features_score(featureVar)
        # sum_var = sum(featureVar)
        # featureScore = featureVar/sum_var
        # return featureScore

    # 假设检验-->卡方检验
    def select_by_chi2(self):
        # 选择K个最好的特征
        # 返回选择特征后的数据
        # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，
        # 回归可用：f_regression(皮尔逊相关系数)，mutual_info_regression(共同信息)
        # 分类可用：chi2(卡方检验)，f_classif(方差分析)，mutual_info_classif(共同信息)
        # 稀疏特征：chi2，mutual_info_regression，mutual_info_classif
        # 输出二元组（评分，P值）的数组，数组第i项为第i个特征的评分和P值。
        # score_func=chi2时，即卡方检验
        selector = SelectKBest(score_func=chi2, k=self.K)
        selector.fit_transform(self.train_X, self.train_y)
        print("scores_:", selector.scores_)
        print("pvalues_:", selector.pvalues_)
        print("selected index:", selector.get_support(True))

        mask = selector.get_support(True)
        feature_names = np.array(self.continuous_feature_names)[mask]
        print("selected name:", feature_names)


    # 相关系数-->皮尔逊相关系数
    def select_by_pearsonr(self):
        # 使用SelectKBest，并自定义平方函数--皮尔逊相关系数
        # 返回特征选择后的数据
        selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=self.K)
        selector.fit_transform(self.train_X, self.train_y)
        # print("scores_:", selector.scores_)
        self.dict_features_score(selector.scores_)
        print("pvalues_:", selector.pvalues_)
        print("selected index:", selector.get_support(True))

        mask = selector.get_support(True)
        feature_names = np.array(self.continuous_feature_names)[mask]
        print("selected name:", feature_names)

    # 互信息-->互信息和最大信息系数
    def select_by_mic(self):
        # 互信息和最大信息系数
        # 返回特征选择后的数据
        # 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
        def mic(x, y):
            m = MINE()
            m.compute_score(x, y)
            return (m.mic(), 0.25)

        # 使用SelectKBest，并自定义平方函数--mic
        selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0], k=self.K)
        f = selector.fit_transform(self.train_X, self.train_y)

        # print("scores_:", selector.scores_)
        self.dict_features_score(selector.scores_)
        print("pvalues_:", selector.pvalues_)
        print("selected index:", selector.get_support(True))

        mask = selector.get_support(True)
        feature_names = np.array(self.continuous_feature_names)[mask]
        print("selected name:", feature_names)


    """
    【Wrapper model(封装器)】-->包裹式选择
    递归式消除特征(RFE):
        将全部特征都丢到给定的模型里面，模型会输出每个特征的重要性，然后删除那些不太重要的特征；
        把剩下的特征再次丢到模型里面，又会输出各个特征的重要性，再次删除；
        如此循环，直至最后剩下目标维度的特征值。
    """
    # 递归特征消除法
    def select_by_RFE(self, models=None):
        # 可选择K个最优特征
        # 向后消除：该过程从所有特征集开始。通过逐步删除集合中剩余的最差特征。
        # 参数estimator为基模型
        # 参数n_features_to_select为选择的特征个数
        selector = RFE(estimator=models, n_features_to_select=self.K)
        selector.fit_transform(self.train_X, self.train_y)
        print("ranking_:", selector.ranking_)
        # print("support_:", selector.support_)

        mask = selector.support_
        feature_names = self.continuous_feature_names
        new_features = []
        for bool, feature in zip(mask, feature_names):
            if bool:
                new_features.append(feature)
        print("selected name:", new_features)

    def select_by_RFECV(self, model=None):
        # 展示：随着特征个数增加得分变化趋势图
        # 向后消除：该过程从所有特征集开始。通过逐步删除集合中剩余的最差特征。
        # 参数estimator为基模型

        selector = RFECV(estimator=model)
        f = selector.fit_transform(self.train_X, self.train_y)
        # grid_scores = list(map(lambda x: round(x, 4), selector.grid_scores_))
        # print("随着特征个数增加，得分变化 : {}".format(grid_scores))

        model_name = str(model).split('(')[0]
        plt.figure()
        plt.title('RFECV of {}'.format(model_name))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        plt.grid()
        plt.show()


    # 前向选择
    def select_by_SFS(self, model=None):
        # 前向选择：该过程从一个空的特性集合开始，并逐个添加最优特征到集合中。
        # 展示：随着特征个数增加得分变化趋势图
        # 可选择K个最优特征
        selector = SFS(model,
                  k_features=self.K,
                  forward=True,
                  floating=False,
                  # scoring='neg_mean_squared_error',
                  cv=0)
        selector.fit(self.train_X, self.train_y)
        k_feature = selector.k_feature_names_
        print('selected features:', k_feature)
        print('selected index:', selector.k_feature_idx_)

        if self.showFig:
            model_name = str(model).split('(')[0]
            plot_sfs(selector.get_metric_dict(), kind='std_dev')
            plt.title('SFS of {}'.format(model_name))
            plt.grid()
            plt.show()


    """
    【Embedded】-->嵌入式选择
    使用SelectFromModel选取特征
        以用来处理任何带有coef_或者feature_importances_ 属性的训练之后的模型。 
        如果相关的coef_ 或者 feature_importances 属性值低于预先设置的阈值，这些特征将会被认为不重要并且移除掉。
        除了指定数值上的阈值之外，还可以通过给定字符串参数来使用内置的启发式方法找到一个合适的阈值。
        可以使用的启发式方法有 mean 、 median 以及使用浮点数乘以这些（例如，0.1*mean ）
        
    可选嵌入交叉验证框架
    
    区别于递归式消除特征:
        该方法不需要重复训练模型，只需要训练一次即可；二是该方法是指定权重的阈值，不是指定特征的维度。
    """
    # 基于线性学习模型的特征排序
    def select_by_linearmodel(self, models=None):
        # Embedded
        '''
        线性模型
        :param models:
        :return:
        '''
        if self.numNull != 0:
            print('特征中有NaN！！！')
        elif self.numInf != 0:
            print('特征中有Inf！！！')
        else:
            if not models:
                models = [LinearRegression(),
                          Ridge(),
                          Lasso()]

            # 使用SelectFromModel训练一次，选择特征
            for model in models:
                model_name = str(model).split('(')[0]
                selector = SelectFromModel(model, max_features=self.K, threshold=-np.inf)
                selector.fit_transform(X=self.train_X, y=self.train_y)
                mask = selector.get_support(True)
                feature_names = np.array(self.continuous_feature_names)[mask]
                print("{} selected feature:{}".format(model_name, feature_names))

            if self.showFig:
                for clf in models:
                    model_name = str(clf).split('(')[0]
                    model = clf.fit(self.train_X, self.train_y)
                    self.dict_features_score(model.coef_)
                    # print(sorted(dict(zip(self.continuous_feature_names, model.coef_)).items(), key=lambda x: x[1], reverse=True))
                    sns.barplot(self.continuous_feature_names, abs(model.coef_))
                    plt.title('{} coef of features'.format(model_name))
                    plt.show()

    # 基于非线性学习模型的特征排序
    def select_by_nonlinearmodel(self, models=None):
        """
        树模型
        :param models:
        :return:
        """
        if not models:
            if (self.numNull != 0) | (self.numInf != 0):
                print('特征中有NaN或Inf！！！')
                print('NaN：{}，Inf：{}'.format(self.numNull, self.numInf))
                models = [XGBRegressor(n_estimators=100, objective='reg:squarederror'),
                      LGBMRegressor(n_estimators=100)]
            else:
                models = [
                      DecisionTreeRegressor(),
                      # RF相对较慢
                      RandomForestRegressor(),
                      GradientBoostingRegressor(),
                      MLPRegressor(solver='lbfgs', max_iter=100),
                      XGBRegressor(n_estimators=100, objective='reg:squarederror'),
                      LGBMRegressor(n_estimators=100)]

        # 使用SelectFromModel训练一次，选择特征
        for model in models:
            model_name = str(model).split('(')[0]
            selector = SelectFromModel(model, max_features=self.K, threshold=-np.inf)
            selector.fit_transform(X=self.train_X, y=self.train_y)
            mask = selector.get_support(True)
            feature_names = np.array(self.continuous_feature_names)[mask]
            print("{} selected feature:{}".format(model_name, feature_names))

        if self.showFig:
            for model in models:
                model_name = str(model).split('(')[0]
                model.fit(self.train_X, self.train_y)

                self.dict_features_score(model.feature_importances_)
                # print(sorted(dict(zip(self.continuous_feature_names, model.feature_importances_)).items(), key=lambda x: x[1], reverse=True))
                sns.barplot(abs(model.feature_importances_), self.continuous_feature_names)
                plt.title('{} importances of features'.format(model_name))
                plt.show()


    """
    置换特征重要性:
        在训练好模型后，对其中一个特征变量进行随机置换顺序。
        如果单个变量的顺序打乱后导致模型预测精度降低，那说明这个变量是重要的。
    """
    def selec_by_features_random_plt(self, model=None):
        if not model:
            mymodel = LinearRegression()
            # mymodel = RandomForestRegressor()
        else:
            mymodel= model
        mymodel.fit(self.train_X, self.train_y)
        result = permutation_importance(mymodel, self.train_X, self.train_y, n_repeats=10,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=self.train_X.columns[sorted_idx])
        ax.set_title("Permutation Importances (test set)")
        fig.tight_layout()
        plt.show()


    def selec_by_features_random_weight(self, model=None):
        train_X, val_X, train_y, val_y = train_test_split(self.train_X, self.train_y, random_state=1)
        if not model:
            mymodel = LinearRegression()
        else:
            mymodel= model


        from eli5.sklearn import PermutationImportance
        perm = PermutationImportance(mymodel, n_iter=5, random_state=1024, cv=5)
        perm.fit(train_X.values, train_y.values)

        result_ = {'var': train_X.columns.values, 'feature_importances_': perm.feature_importances_,
                   'feature_importances_std_': perm.feature_importances_std_}
        feature_importances_ = pd.DataFrame(result_, columns=['var', 'feature_importances_', 'feature_importances_std_'])
        feature_importances_ = feature_importances_.sort_values('feature_importances_', ascending=False)

        # import eli5
        # display(eli5.show_weights(perm))
        # eli5.show_weights(perm, feature_names=train_X.columns.tolist()) #结果可视化

        sel = SelectFromModel(perm, threshold=0.00, prefit=True)
        X_train_ = sel.transform(train_X)
        X_valid_ = sel.transform(val_X)

        return feature_importances_, X_train_, X_valid_
