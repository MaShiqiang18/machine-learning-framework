# coding=gbk

import os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from trainModels.untils import defindMetrics
import joblib
import pickle


class loadTreeModel(object):
    def __init__(self, modelName, config):
        self.modelName = modelName
        self.config = config
        self.modelType = config.modelType
        self.savePath = config.path_data.saveModelPath

    def save_model(self, clf, save_type):
        if save_type == 'J':
            filePath = self.savePath + '/joblib_models/%s/' % self.config.modelType
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            joblib.dump(clf, filePath + self.modelName + '.pkl')
            print('已保存当前模型：{}'.format(self.modelName + '.pkl'))
        elif save_type == 'P':
            filePath = self.savePath + '/pickle_models/%s/' % self.config.modelType
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            fileName = filePath + self.modelName + '.txt'
            with open(fileName, 'wb') as f:
                pickle.dump(clf, f)
            print('已保存当前模型：{}'.format(self.modelName + '.txt'))
        else:
            print('不保存当前模型：{}'.format(self.modelName))

    def load_model(self):
        joblib_models_path = self.savePath + '/joblib_models/%s/' % self.config.modelType + self.modelName + '.pkl'
        pickle_models_path = self.savePath + '/pickle_models/%s/' % self.config.modelType + self.modelName + '.txt'

        # 调用已有模型，进行增量训练
        if os.path.exists(joblib_models_path):
            print('调用模型：{}，进行增量训练'.format(self.modelName + '.pkl'))
            model_load = joblib.load(joblib_models_path)
            return model_load

        elif os.path.exists(pickle_models_path):
            print('调用模型：{}，进行增量训练'.format(self.modelName + '.txt'))
            with open(pickle_models_path, 'rb') as f:
                model_load = pickle.load(f)
            return model_load

        else:
            return None

    def GBDT_train(self,X_train, X_valid, labels_train, labels_valid, X_test, gbdt_params_all):
        gbdt_params = gbdt_params_all.copy()
        objective_type = gbdt_params['objective_type']
        gbdt_params.pop('objective_type')
        if not self.config.retrain:
            # 调用已有模型进行增量训练
            model_load = self.load_model()
            if not model_load:
                print('不存在模型：{}，从头训练'.format(self.modelName))
                if objective_type == 'regressor':
                    clf = GradientBoostingRegressor(**gbdt_params)
                else:
                    clf = GradientBoostingClassifier(**gbdt_params)
                clf.fit(X_train, labels_train)
            else:
                clf = model_load.fit(X_train, labels_train)
        else:
            if objective_type == 'regressor':
                clf = GradientBoostingRegressor(**gbdt_params)
            else:
                clf = GradientBoostingClassifier(**gbdt_params)
            clf.fit(X_train, labels_train)

        val_xgb_pre = clf.predict(X_valid.values)
        test_xgb_pre = clf.predict(X_test.values)

        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        score_xgb = myMetrics.metricsFunc(val_xgb_pre, labels_valid)
        self.save_model(clf, self.config.saveModel)
        return val_xgb_pre, test_xgb_pre, score_xgb


    def LGB_train(self,X_train, X_valid, labels_train, labels_valid, X_test, lgb_param_all):
        lgb_param_contrl = {'early_stopping_rounds': 100, 'categorical_feature': 'auto'}
        lgb_param = lgb_param_all.copy()
        objective_type = lgb_param['objective_type']
        lgb_param.pop('objective_type')

        for k in ['early_stopping_rounds', 'categorical_feature']:
            if k in lgb_param:
                lgb_param_contrl[k] = lgb_param[k]
                lgb_param.pop(k)

        if not self.config.retrain:
            # 调用已有模型进行增量训练
            model_load = self.load_model()
            if not model_load:
                print('不存在模型：{}，从头训练'.format(self.modelName))
                if objective_type == 'regressor':
                    clf = LGBMRegressor(**lgb_param)
                else:
                    clf = LGBMClassifier(**lgb_param)

                clf.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], eval_metric='rmse',
                        early_stopping_rounds=lgb_param_contrl['early_stopping_rounds'],
                        categorical_feature=lgb_param_contrl['categorical_feature'])
            else:
                clf = model_load.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], eval_metric='rmse',
                                     early_stopping_rounds=lgb_param_contrl['early_stopping_rounds'],
                                     categorical_feature=lgb_param_contrl['categorical_feature'])
        else:
            if objective_type == 'regressor':
                clf = LGBMRegressor(**lgb_param)
            else:
                clf = LGBMClassifier(**lgb_param)
            clf.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], eval_metric='rmse',
                    early_stopping_rounds=lgb_param_contrl['early_stopping_rounds'],
                    categorical_feature=lgb_param_contrl['categorical_feature'])


        val_lgb_pre = clf.predict(X_valid.values, num_iteration=clf.best_iteration_)
        test_lgb_pre = clf.predict(X_test.values, num_iteration=clf.best_iteration_)

        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        score_lgb = myMetrics.metricsFunc(val_lgb_pre, labels_valid)

        self.save_model(clf, self.config.saveModel)
        return val_lgb_pre, test_lgb_pre, score_lgb


    def XGB_train(self,X_train, X_valid, labels_train, labels_valid, X_test, xgb_params_all):
        xgb_param_contrl = {'early_stopping_rounds': 100}
        xgb_params = xgb_params_all.copy()
        objective_type = xgb_params['objective_type']
        xgb_params.pop('objective_type')

        for k in xgb_param_contrl.keys():
            if k in xgb_params:
                xgb_param_contrl[k] = xgb_params[k]
                xgb_params.pop(k)

        if not self.config.retrain:
            # 调用已有模型进行增量训练
            model_load = self.load_model()
            if not model_load:
                print('不存在模型：{}，从头训练'.format(self.modelName))
                if objective_type == 'regressor':
                    clf = XGBRegressor(**xgb_params)
                else:
                    clf = XGBClassifier(**xgb_params)
                clf.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], eval_metric='rmse',
                        early_stopping_rounds=xgb_param_contrl['early_stopping_rounds'])
            else:
                clf = model_load.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], eval_metric='rmse',
                        early_stopping_rounds=xgb_param_contrl['early_stopping_rounds'])
        else:
            if objective_type == 'regressor':
                clf = XGBRegressor(**xgb_params)
            else:
                clf = XGBClassifier(**xgb_params)


            clf.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], eval_metric='rmse',
                    early_stopping_rounds=xgb_param_contrl['early_stopping_rounds'])

        val_xgb_pre = clf.predict(X_valid, ntree_limit=clf.best_iteration)
        test_xgb_pre = clf.predict(X_test, ntree_limit=clf.best_iteration)

        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        score_xgb = myMetrics.metricsFunc(val_xgb_pre, labels_valid)
        self.save_model(clf, self.config.saveModel)
        return val_xgb_pre, test_xgb_pre, score_xgb


    def CGB_train(self,X_train, X_valid, labels_train, labels_valid, X_test, cgb_params_all):
        cgb_params = cgb_params_all.copy()
        objective_type = cgb_params['objective_type']
        cgb_params.pop('objective_type')

        cgb_param_contrl = {'verbose': 200, 'early_stopping_rounds': 100}
        for k in cgb_param_contrl.keys():
            if k in cgb_params:
                cgb_param_contrl[k] = cgb_params[k]
                cgb_params.pop(k)
        if 'cat_features' in cgb_params:
            cgb_param_contrl['cat_features'] = cgb_params['cat_features']
            cgb_params.pop('cat_features')
        else:
            cgb_param_contrl['cat_features'] = None

        if not self.config.retrain:
            # 调用已有模型进行增量训练
            model_load = self.load_model()
            if not model_load:
                print('不存在模型：{}，从头训练'.format(self.modelName))
                if objective_type == 'regressor':
                    clf = CatBoostRegressor(**cgb_params)
                else:
                    clf = CatBoostClassifier(**cgb_params)
                clf.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], verbose=cgb_param_contrl['verbose'],
                        early_stopping_rounds=cgb_param_contrl['early_stopping_rounds'],
                        cat_features=cgb_param_contrl['cat_features'])
            else:
                clf = model_load.fit(X_train, labels_train,
                                     eval_set=[(X_valid, labels_valid)],
                                     verbose=cgb_param_contrl['verbose'],
                                     early_stopping_rounds=cgb_param_contrl['early_stopping_rounds'],
                                     cat_features=cgb_param_contrl['cat_features'])

        else:
            if objective_type == 'regressor':
                clf = CatBoostRegressor(**cgb_params)
            else:
                clf = CatBoostClassifier(**cgb_params)

            clf.fit(X_train, labels_train, eval_set=[(X_valid, labels_valid)], verbose=cgb_param_contrl['verbose'],
                    early_stopping_rounds=cgb_param_contrl['early_stopping_rounds'],
                    cat_features=cgb_param_contrl['cat_features'])

        val_xgb_pre = clf.predict(X_valid, ntree_end=clf.best_iteration_)
        test_xgb_pre = clf.predict(X_test, ntree_end=clf.best_iteration_)

        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        score_xgb = myMetrics.metricsFunc(val_xgb_pre, labels_valid)
        self.save_model(clf, self.config.saveModel)
        return val_xgb_pre, test_xgb_pre, score_xgb


    def train(self, X_train, X_valid, labels_train, labels_valid, X_test, param):
        if self.modelType == 'GBDT':
            val_gbdt_pre, test_gbdt_pre, score_gbdt = self.GBDT_train(X_train, X_valid, labels_train, labels_valid,
                                                                  X_test, param)
            return val_gbdt_pre, test_gbdt_pre, score_gbdt
        elif self.modelType == 'LGB':
            val_lgb_pre, test_lgb_pre, score_lgb = self.LGB_train(X_train, X_valid, labels_train, labels_valid,
                                                                  X_test, param)
            return val_lgb_pre, test_lgb_pre, score_lgb
        elif self.modelType == 'XGB':
            val_xgb_pre, test_xgb_pre, score_xgb = self.XGB_train(X_train, X_valid, labels_train, labels_valid,
                                                                  X_test, param)
            return val_xgb_pre, test_xgb_pre, score_xgb
        elif self.modelType == 'CGB':
            val_cgb_pre, test_cgb_pre, score_cgb = self.CGB_train(X_train, X_valid, labels_train, labels_valid,
                                                                  X_test, param)
            return val_cgb_pre, test_cgb_pre, score_cgb
        else:
            print('请确认模型名称')
            return None

