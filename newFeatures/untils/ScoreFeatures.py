# coding=gbk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
# ��ģ��
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost.sklearn import XGBRegressor, XGBClassifier
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
# ����ʽѡ��
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import RFE, RFECV
# ����ʽѡ��
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile
# Ƕ��ʽѡ��
from sklearn.feature_selection import SelectFromModel
# ��������
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
# import eli5
# from eli5.sklearn import PermutationImportance

# ���ۺ���
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy.stats import pearsonr
from minepy import MINE


import warnings

warnings.simplefilter("ignore")


class Score_of_features(object):
    """
    ʹ��Filter��Wrapper��Embedded�Լ�ͨ��������ֵ�����ģʽ������������Ҫ��
    """
    def __init__(self, data, continuous_feature_names, label, score, t='R', showFig=False):
        self.data = data
        self.continuous_feature_names = continuous_feature_names
        self.label = label
        self.score = score
        self.K = len(continuous_feature_names)
        self.T = t
        self.showFig = showFig
        # ��ѡ����
        self.train_X = data[continuous_feature_names]
        self.train_y = data[label]
        self.numNull = self.train_X.isnull().sum().sum()
        self.numInf = np.isinf(self.train_X.values).sum()

        # ��ѡģ��
        self.linearRegressionModel = [LinearRegression(), Ridge(), Lasso(), LinearSVR()]
        self.linearClassModel = [LogisticRegression(), LinearSVC(), RidgeClassifier()]

        self.treeRegressionModel = [ExtraTreesRegressor(),
                                    DecisionTreeRegressor(),
                                    RandomForestRegressor(),  # RF��Խ���
                                    GradientBoostingRegressor(),
                                    XGBRegressor(n_estimators=100, objective='reg:squarederror'),
                                    LGBMRegressor(n_estimators=100)]
        self.treeClassModel = [ExtraTreesClassifier(),
                               DecisionTreeClassifier(),
                               RandomForestClassifier(),
                               GradientBoostingClassifier(),
                               XGBClassifier(n_estimators=100, objective="binary:logistic"),
                               LGBMClassifier(n_estimators=100)]

        self.nonlinearRegressionModel = self.treeRegressionModel + [SVR(), MLPRegressor(solver='lbfgs', max_iter=100),]
        self.nonlinearClassModel = self.treeClassModel + [SVC(), MLPClassifier(),]

    """
    ��Filter model(������)��-->����ʽѡ��
    ����������:
        ���ڵ�һ������Ŀ��y֮��Ĺ�ϵ��ͨ������ĳ���ܹ�����������Ҫ�Ե�ָ�꣬Ȼ��ѡ����Ҫ��Top��K��������
    ȱ��: ������������ϵ����
    """

    # ����ѡ��(�Ƴ��ͷ�������)
    def score_of_var(self):
        # �Ƴ���Щ�������ĳ����ֵ��������ֵ�䶯����С��ĳ����Χ����������һ�������������ֶȽϲ�����Ƴ���
        # ����ֵΪ����ѡ��������
        # ����thresholdΪ�������ֵ
        selector = VarianceThreshold()
        selector.fit_transform(self.train_X)

        sc = [abs(x) for x in selector.variances_]
        sum_sc = sum(sc)
        featureScore = [round(s/sum_sc, 4) for s in sc]

        print('Get variance score')
        return featureScore

    # �������-->��������
    def score_of_chi2(self):
        # ѡ��K����õ�����
        # ����ѡ�������������
        # ��һ������Ϊ�������������Ƿ�õĺ������ú����������������Ŀ��������
        # �ع���ã�f_regression(Ƥ��ѷ���ϵ��)��mutual_info_regression(��ͬ��Ϣ)
        # ������ã�chi2(��������)��f_classif(�������)��mutual_info_classif(��ͬ��Ϣ)
        # ϡ��������chi2��mutual_info_regression��mutual_info_classif
        # �����Ԫ�飨���֣�Pֵ�������飬�����i��Ϊ��i�����������ֺ�Pֵ��
        # score_func=chi2ʱ������������
        selector = SelectKBest(score_func=chi2, k=self.K)
        selector.fit_transform(self.train_X, self.train_y)

        # print("scores_:", selector.scores_)
        # print("pvalues_:", selector.pvalues_)

        sc = [abs(x) for x in selector.scores_]
        sum_sc = sum(sc)
        featureScore = [round(s/sum_sc, 4) for s in sc]
        print('Get chi2 score')
        return featureScore

    # ���ϵ��-->Ƥ��ѷ���ϵ��
    def score_of_pearsonr(self):
        # ʹ��SelectKBest�����Զ���ƽ������--Ƥ��ѷ���ϵ��
        # ��������ѡ��������
        selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=self.K)
        selector.fit_transform(self.train_X, self.train_y)
        # print("scores_:", selector.scores_)
        # print("pvalues_:", selector.pvalues_)

        sc = [abs(x) for x in selector.scores_]
        sum_sc = sum(sc)
        featureScore = [round(s/sum_sc, 4) for s in sc]
        print('Get pearsonr score')
        return featureScore


    # ����Ϣ-->����Ϣ�������Ϣϵ��
    def score_of_mic(self):
        # ����Ϣ�������Ϣϵ��
        # ��������ѡ��������
        # ����MINE����Ʋ��Ǻ���ʽ�ģ�����mic��������Ϊ����ʽ�ģ�����һ����Ԫ�飬��Ԫ��ĵ�2�����óɹ̶���Pֵ0.5
        def mic(x, y):
            m = MINE()
            m.compute_score(x, y)
            return (m.mic(), 0.25)

        # ʹ��SelectKBest�����Զ���ƽ������--mic
        selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: mic(x, Y), X.T))).T[0], k=self.K)
        selector.fit_transform(self.train_X, self.train_y)
        # print("scores_:", selector.scores_)
        # print("pvalues_:", selector.pvalues_)

        sc = [abs(x) for x in selector.scores_]
        sum_sc = sum(sc)
        featureScore = [round(s/sum_sc, 4) for s in sc]
        print('Get mic score')
        return featureScore

    """
    ��Wrapper model(��װ��)��-->����ʽѡ��
    �ݹ�ʽ��������(RFE):
        ��ȫ������������������ģ�����棬ģ�ͻ����ÿ����������Ҫ�ԣ�Ȼ��ɾ����Щ��̫��Ҫ��������
        ��ʣ�µ������ٴζ���ģ�����棬�ֻ����������������Ҫ�ԣ��ٴ�ɾ����
        ���ѭ����ֱ�����ʣ��Ŀ��ά�ȵ�����ֵ��
    """

    # �ݹ�����������
    def score_of_RFE(self, model=None):
        # ��ѡ��K����������
        # ����������ù��̴�������������ʼ��ͨ����ɾ��������ʣ������������
        # ����estimatorΪ��ģ��
        # ����n_features_to_selectΪѡ�����������
        selector = RFE(estimator=model, n_features_to_select=self.K)
        f = selector.fit_transform(self.train_X, self.train_y)

        # print("ranking_:", selector.ranking_)

        sc = [1/abs(x) for x in selector.ranking_]
        sum_sc = sum(sc)
        featureScore = [round(s/sum_sc, 4) for s in sc]

        model_name = str(model).split('(')[0]
        print(model_name + ' by RFE is finished')
        return featureScore

    def figs_of_RFE(self, model=None):
        # չʾ�����������������ӵ÷ֱ仯����ͼ
        # ����������ù��̴�������������ʼ��ͨ����ɾ��������ʣ������������
        # ����estimatorΪ��ģ��

        selector = RFECV(estimator=model, scoring=self.score)
        selector.fit_transform(self.train_X, self.train_y)

        model_name = str(model).split('(')[0]
        plt.figure()
        plt.title('RFECV of {}'.format(model_name))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        plt.grid()
        plt.show()

    # ǰ��ѡ��
    def score_of_SFS(self, model=None):
        # ǰ��ѡ�񣺸ù��̴�һ���յ����Լ��Ͽ�ʼ�������������������������С�
        # չʾ�����������������ӵ÷ֱ仯����ͼ
        # ��ѡ��K����������
        selector = SFS(model,
                       k_features=self.K,
                       forward=True,
                       floating=False,
                       # scoring='neg_mean_squared_error',
                       scoring=self.score,
                       cv=0)
        selector.fit(self.train_X, self.train_y)

        features_idx = []
        for k, v in selector.get_metric_dict().items():
            for f in v['feature_idx']:
                if f not in features_idx:
                    # ��˳��ȡ����Ҫ����ߵ�����
                    features_idx.append(f)

        sort_num = []
        for f in self.continuous_feature_names:
            i = features_idx.index(f)+1
            sort_num.append(i)

        sc = [1/x for x in sort_num]
        sum_sc = sum(sc)
        featureScore = [round(s/sum_sc, 4) for s in sc]

        model_name = str(model).split('(')[0]
        print(model_name + ' by SFS is finished')
        return featureScore

    def figs_of_SFS(self, model=None):
        selector = SFS(model,
                       k_features=self.K,
                       forward=True,
                       floating=False,
                       # scoring='neg_mean_squared_error',
                       cv=0)
        selector.fit(self.train_X, self.train_y)
        model_name = str(model).split('(')[0]
        fig1 = plot_sfs(selector.get_metric_dict(), kind='std_dev')
        plt.title('SFS of {}'.format(model_name))
        plt.grid()
        plt.show()

    """
    ��Embedded��-->Ƕ��ʽѡ��
    ʹ��SelectFromModelѡȡ����
        �����������κδ���coef_����feature_importances_ ���Ե�ѵ��֮���ģ�͡� 
        �����ص�coef_ ���� feature_importances ����ֵ����Ԥ�����õ���ֵ����Щ�������ᱻ��Ϊ����Ҫ�����Ƴ�����
        ����ָ����ֵ�ϵ���ֵ֮�⣬������ͨ�������ַ���������ʹ�����õ�����ʽ�����ҵ�һ�����ʵ���ֵ��
        ����ʹ�õ�����ʽ������ mean �� median �Լ�ʹ�ø�����������Щ�����磬0.1*mean ��

    ��ѡǶ�뽻����֤���

    �����ڵݹ�ʽ��������:
        �÷�������Ҫ�ظ�ѵ��ģ�ͣ�ֻ��Ҫѵ��һ�μ��ɣ����Ǹ÷�����ָ��Ȩ�ص���ֵ������ָ��������ά�ȡ�
    """

    # ��������ѧϰģ�͵���������
    def score_of_linearmodel(self, model=None):
        # Embedded
        '''
        ����ģ��
        :param models:
        :return:
        '''
        if self.numNull != 0:
            print('��������NaN������')
        elif self.numInf != 0:
            print('��������Inf������')
        else:
            if not model:
                model = LinearRegression()

            model_name = str(model).split('(')[0]
            model.fit(self.train_X, self.train_y)

            if self.showFig:
                sns.barplot(self.continuous_feature_names, abs(model.coef_))
                plt.title('{} coef of features'.format(model_name))
                plt.show()

            sc = [abs(x) for x in model.coef_]
            sum_sc = sum(sc)
            featureScore = [round(s / sum_sc, 4) for s in sc]
            print(model_name + ' is finished')

            return featureScore

    # ���ڷ�����ѧϰģ�͵���������
    def score_of_nonlinearmodel(self, model=None):
        """
        ��ģ��
        :param models:
        :return:
        """
        if not [model]:
            if (self.numNull != 0) | (self.numInf != 0):
                print('��������NaN��Inf������')
                print('NaN��{}��Inf��{}'.format(self.numNull, self.numInf))
            model = LGBMRegressor(n_estimators=100)

        model_name = str(model).split('(')[0]
        model.fit(self.train_X, self.train_y)

        if self.showFig:
            sns.barplot(abs(model.feature_importances_), self.continuous_feature_names)
            plt.title('{} importances of features'.format(model_name))
            plt.show()

        sc = [abs(x) for x in model.feature_importances_]
        sum_sc = sum(sc)
        featureScore = [round(s / sum_sc, 4) for s in sc]
        print(model_name + ' is finished')

        return featureScore


    """
    �û�������Ҫ��:
        ��ѵ����ģ�ͺ󣬶�����һ������������������û�˳��
        �������������˳����Һ���ģ��Ԥ�⾫�Ƚ��ͣ���˵�������������Ҫ�ġ�
    """

    def score_of_features_random_plt(self):
        rf = RandomForestRegressor()
        rf.fit(self.train_X, self.train_y)
        result = permutation_importance(rf, self.train_X, self.train_y, n_repeats=10,
                                        random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()

        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T,
                   vert=False, labels=self.train_X.columns[sorted_idx])
        ax.set_title("Permutation Importances (test set)")
        fig.tight_layout()
        plt.show()

    # def selec_by_features_random_weight(self, model=None):
    #     train_X, val_X, train_y, val_y = train_test_split(self.train_X, self.train_y, random_state=1)
    #     if not model:
    #         mymodel = LinearRegression()
    #     else:
    #         mymodel= model
    #
    #     from eli5.sklearn import PermutationImportance
    #     perm = PermutationImportance(mymodel, n_iter=5, random_state=1024, cv=5)
    #     perm.fit(train_X.values, train_y.values)
    #
    #     sc = [abs(x) for x in perm.feature_importances_]
    #     sum_sc = sum(sc)
    #     featureScore = [round(s / sum_sc, 4) for s in sc]
    #     model_name = str(model).split('(')[0]
    #     print(model_name + 'by random features is finished')
    #
    #     return featureScore

    def score_of_Filter(self):
        result = pd.DataFrame(index=self.continuous_feature_names)
        varScore = self.score_of_var()
        result['Variance'] = varScore
        if self.T == 'R':
            pScore = self.score_of_pearsonr()
            result['Pearsonr'] = pScore
            mScore = self.score_of_mic()
            result['Mic'] = mScore
        else:
            cScore = self.score_of_chi2()
            result['Chi2'] = cScore
        return result

    def score_of_Wrapper(self, models=None):
        result = pd.DataFrame(index=self.continuous_feature_names)
        if self.T == 'R':
            if not models:
                models = self.linearRegressionModel + self.nonlinearRegressionModel
            for m in models:
                model_name = str(m).split('(')[0]
                msScore = self.score_of_SFS(m)
                result['SFS_' + model_name] = msScore

                mrScore = self.score_of_RFE(m)
                result['RFE_' + model_name] = mrScore

        else:
            if not models:
                models = self.linearClassModel + self.nonlinearClassModel
            for m in models:
                model_name = str(m).split('(')[0]
                msScore = self.score_of_SFS(m)
                result['SFS_' + model_name] = msScore

                mrScore = self.score_of_RFE(m)
                result['RFE_' + model_name] = mrScore

        return result

    def score_of_Embedded(self, models=None):
        result = pd.DataFrame(index=self.continuous_feature_names)
        if self.T == 'R':
            if not models:
                models = self.linearRegressionModel + self.nonlinearRegressionModel
            for m in models:
                model_name = str(m).split('(')[0]
                linearRegressionModels = [str(model).split('(')[0] for model in self.linearRegressionModel]
                treeRegressionModels = [str(model).split('(')[0] for model in self.treeRegressionModel]
                if model_name in linearRegressionModels:
                    mlmScore = self.score_of_linearmodel(m)
                    result['Embedded_' + model_name] = mlmScore
                elif model_name in treeRegressionModels:
                    mnlmScore = self.score_of_nonlinearmodel(m)
                    result['Embedded_' + model_name] = mnlmScore
                else:
                    pass

        else:
            if not models:
                models = self.linearClassModel + self.nonlinearClassModel
            for m in models:
                model_name = str(m).split('(')[0]
                linearClassModels = [str(model).split('(')[0] for model in self.linearClassModel]
                treeClassModels = [str(model).split('(')[0] for model in self.treeClassModel]
                if model_name in linearClassModels:
                    mlmScore = self.score_of_linearmodel(m)
                    result['Embedded_' + model_name] = mlmScore
                elif model_name in treeClassModels:
                    mnlmScore = self.score_of_nonlinearmodel(m)
                    result['Embedded_' + model_name] = mnlmScore
                else:
                    pass

        return result


    def score_main(self, Wrapper_models=None, Embedded_models=None):
        result = pd.DataFrame()
        print('\nFilter start...')
        result_Filter = self.score_of_Filter()
        print('=====> Filter scores:', result_Filter.shape[1])
        result = pd.concat([result, result_Filter], axis=1)
        # print('\nWrapper start...')
        # result_Wrapper = self.score_of_Wrapper(models=Wrapper_models)
        # print('=====> Wrapper scores:', result_Wrapper.shape[1])
        # result = pd.concat([result, result_Wrapper], axis=1)
        print('\nEmbedded start...')
        result_Embedded = self.score_of_Embedded(models=Embedded_models)
        print('=====> Embedded scores:', result_Embedded.shape[1])
        result = pd.concat([result, result_Embedded], axis=1)

        result['mean score'] = result.mean(axis=1)
        result['median score'] = result.median(axis=1)
        result['max score'] = result.max(axis=1)

        for i in range(result.shape[1]):
            if i < result.shape[1] - 3:
                plt.plot(range(result.shape[0]), result.iloc[:, i], label=result.columns[i])
            else:
                plt.plot(range(result.shape[0]), result.iloc[:, i], linewidth=3, label=result.columns[i])
        plt.legend()
        plt.show()

        result['mean score'] = result.mean(axis=1)
        result['median score'] = result.median(axis=1)
        result['max score'] = result.max(axis=1)

        return result

