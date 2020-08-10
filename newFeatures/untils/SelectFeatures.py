# coding=gbk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
# ��ģ��
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
from IPython.display import display

# ���ۺ���
from sklearn.metrics import mean_absolute_error,  make_scorer
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.feature_selection import f_regression, mutual_info_regression
from scipy.stats import pearsonr
from minepy import MINE

import warnings
warnings.simplefilter("ignore")


class Select_features(object):
    """
    ʹ��Filter��Wrapper��Embedded�Լ�ͨ��������ֵ�����ģʽ��ѡ����õ�K������
    """
    def __init__(self, data, continuous_feature_names, label, score, k=5, c='R', showFig=True):
        self.data = data
        self.continuous_feature_names = continuous_feature_names
        self.label = label
        self.score = score
        self.K = k
        self.C = c
        self.showFig = showFig
        # ��ѡ����
        self.train_X = data[continuous_feature_names]
        self.train_y = data[label]
        self.numNull = self.train_X.isnull().sum().sum()
        self.numInf = np.isinf(self.train_X.values).sum()


    def dict_features_score(self,scores):
        scores = list(map(lambda x: round(x, 4), scores))
        print(sorted(dict(zip(self.continuous_feature_names, scores)).items(), key=lambda x: x[1], reverse=True))

    """
    ��Filter model(������)��-->����ʽѡ��[����������]
    ����������:
        ���ڵ�һ������Ŀ��y֮��Ĺ�ϵ��ͨ������ĳ���ܹ�����������Ҫ�Ե�ָ�꣬Ȼ��ѡ����Ҫ��Top��K��������
    ȱ��: ������������ϵ����
    """
    # ����ѡ��(�Ƴ��ͷ�������)
    def select_by_var(self, threshold=0.):
        # �Ƴ���Щ�������ĳ����ֵ��������ֵ�䶯����С��ĳ����Χ����������һ�������������ֶȽϲ�����Ƴ���
        # ����ֵΪ����ѡ��������
        # ����thresholdΪ�������ֵ
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.train_X)
        data = selector.transform(self.train_X)

        featureVar = selector.variances_
        # print(featureVar)
        self.dict_features_score(featureVar)
        # sum_var = sum(featureVar)
        # featureScore = featureVar/sum_var
        # return featureScore

    # �������-->��������
    def select_by_chi2(self):
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
        print("scores_:", selector.scores_)
        print("pvalues_:", selector.pvalues_)
        print("selected index:", selector.get_support(True))

        mask = selector.get_support(True)
        feature_names = np.array(self.continuous_feature_names)[mask]
        print("selected name:", feature_names)


    # ���ϵ��-->Ƥ��ѷ���ϵ��
    def select_by_pearsonr(self):
        # ʹ��SelectKBest�����Զ���ƽ������--Ƥ��ѷ���ϵ��
        # ��������ѡ��������
        selector = SelectKBest(lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=self.K)
        selector.fit_transform(self.train_X, self.train_y)
        # print("scores_:", selector.scores_)
        self.dict_features_score(selector.scores_)
        print("pvalues_:", selector.pvalues_)
        print("selected index:", selector.get_support(True))

        mask = selector.get_support(True)
        feature_names = np.array(self.continuous_feature_names)[mask]
        print("selected name:", feature_names)

    # ����Ϣ-->����Ϣ�������Ϣϵ��
    def select_by_mic(self):
        # ����Ϣ�������Ϣϵ��
        # ��������ѡ��������
        # ����MINE����Ʋ��Ǻ���ʽ�ģ�����mic��������Ϊ����ʽ�ģ�����һ����Ԫ�飬��Ԫ��ĵ�2�����óɹ̶���Pֵ0.5
        def mic(x, y):
            m = MINE()
            m.compute_score(x, y)
            return (m.mic(), 0.25)

        # ʹ��SelectKBest�����Զ���ƽ������--mic
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
    ��Wrapper model(��װ��)��-->����ʽѡ��
    �ݹ�ʽ��������(RFE):
        ��ȫ������������������ģ�����棬ģ�ͻ����ÿ����������Ҫ�ԣ�Ȼ��ɾ����Щ��̫��Ҫ��������
        ��ʣ�µ������ٴζ���ģ�����棬�ֻ����������������Ҫ�ԣ��ٴ�ɾ����
        ���ѭ����ֱ�����ʣ��Ŀ��ά�ȵ�����ֵ��
    """
    # �ݹ�����������
    def select_by_RFE(self, models=None):
        # ��ѡ��K����������
        # ����������ù��̴�������������ʼ��ͨ����ɾ��������ʣ������������
        # ����estimatorΪ��ģ��
        # ����n_features_to_selectΪѡ�����������
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
        # չʾ�����������������ӵ÷ֱ仯����ͼ
        # ����������ù��̴�������������ʼ��ͨ����ɾ��������ʣ������������
        # ����estimatorΪ��ģ��

        selector = RFECV(estimator=model)
        f = selector.fit_transform(self.train_X, self.train_y)
        # grid_scores = list(map(lambda x: round(x, 4), selector.grid_scores_))
        # print("���������������ӣ��÷ֱ仯 : {}".format(grid_scores))

        model_name = str(model).split('(')[0]
        plt.figure()
        plt.title('RFECV of {}'.format(model_name))
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
        plt.grid()
        plt.show()


    # ǰ��ѡ��
    def select_by_SFS(self, model=None):
        # ǰ��ѡ�񣺸ù��̴�һ���յ����Լ��Ͽ�ʼ�������������������������С�
        # չʾ�����������������ӵ÷ֱ仯����ͼ
        # ��ѡ��K����������
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
    def select_by_linearmodel(self, models=None):
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
            if not models:
                models = [LinearRegression(),
                          Ridge(),
                          Lasso()]

            # ʹ��SelectFromModelѵ��һ�Σ�ѡ������
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

    # ���ڷ�����ѧϰģ�͵���������
    def select_by_nonlinearmodel(self, models=None):
        """
        ��ģ��
        :param models:
        :return:
        """
        if not models:
            if (self.numNull != 0) | (self.numInf != 0):
                print('��������NaN��Inf������')
                print('NaN��{}��Inf��{}'.format(self.numNull, self.numInf))
                models = [XGBRegressor(n_estimators=100, objective='reg:squarederror'),
                      LGBMRegressor(n_estimators=100)]
            else:
                models = [
                      DecisionTreeRegressor(),
                      # RF��Խ���
                      RandomForestRegressor(),
                      GradientBoostingRegressor(),
                      MLPRegressor(solver='lbfgs', max_iter=100),
                      XGBRegressor(n_estimators=100, objective='reg:squarederror'),
                      LGBMRegressor(n_estimators=100)]

        # ʹ��SelectFromModelѵ��һ�Σ�ѡ������
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
    �û�������Ҫ��:
        ��ѵ����ģ�ͺ󣬶�����һ������������������û�˳��
        �������������˳����Һ���ģ��Ԥ�⾫�Ƚ��ͣ���˵�������������Ҫ�ġ�
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
        # eli5.show_weights(perm, feature_names=train_X.columns.tolist()) #������ӻ�

        sel = SelectFromModel(perm, threshold=0.00, prefit=True)
        X_train_ = sel.transform(train_X)
        X_valid_ = sel.transform(val_X)

        return feature_importances_, X_train_, X_valid_
