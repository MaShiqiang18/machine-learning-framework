# coding=gbk

import os
import lightgbm as lgb
import xgboost as xgb
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


    def LGB_train(self,X_train, X_valid, labels_train, labels_valid, X_test, lgb_param, retrain):
        trn_data = lgb.Dataset(X_train.values, labels_train)
        val_data = lgb.Dataset(X_valid.values, labels_valid)

        if not retrain:
            # 调用已有模型进行增量训练
            model_load = self.load_model()
            if not model_load:
                print('不存在模型：{}，从头训练'.format(self.modelName))
                clf = lgb.train(lgb_param,
                        trn_data,
                        1000,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=200,
                        early_stopping_rounds=100)
            else:
                clf = model_load.train(lgb_param,
                        trn_data,
                        1000,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=200,
                        early_stopping_rounds=100)
        else:
            clf = lgb.train(lgb_param,
                        trn_data,
                        1000,
                        valid_sets=[trn_data, val_data],
                        verbose_eval=200,
                        early_stopping_rounds=100)


        val_lgb_pre = clf.predict(X_valid.values, num_iteration=clf.best_iteration)
        test_lgb_pre = clf.predict(X_test.values, num_iteration=clf.best_iteration)

        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        score_lgb = myMetrics.metricsFunc(val_lgb_pre, labels_valid)

        self.save_model(clf, self.config.saveModel)
        return val_lgb_pre, test_lgb_pre, score_lgb


    def XGB_train(self,X_train, X_valid, labels_train, labels_valid, X_test, xgb_params, retrain):
        trn_data = xgb.DMatrix(X_train.values, labels_train)
        val_data = xgb.DMatrix(X_valid.values, labels_valid)

        watchlist = [(trn_data, 'train'), (val_data, 'valid_data')]

        if not retrain:
            # 调用已有模型进行增量训练
            model_load = self.load_model()
            if not model_load:
                print('不存在模型：{}，从头训练'.format(self.modelName))
                clf = xgb.train(dtrain=trn_data, num_boost_round=10, evals=watchlist, early_stopping_rounds=20,
                                verbose_eval=20, params=xgb_params)
            else:
                clf = model_load.train(dtrain=trn_data, num_boost_round=10, evals=watchlist, early_stopping_rounds=20,
                                verbose_eval=20, params=xgb_params)

        else:
            clf = xgb.train(dtrain=trn_data, num_boost_round=10, evals=watchlist, early_stopping_rounds=20,
                            verbose_eval=20, params=xgb_params)

        val_xgb_pre = clf.predict(xgb.DMatrix(X_valid.values), ntree_limit=clf.best_ntree_limit)
        test_xgb_pre = clf.predict(xgb.DMatrix(X_test.values), ntree_limit=clf.best_ntree_limit)

        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        score_xgb = myMetrics.metricsFunc(val_xgb_pre, labels_valid)
        self.save_model(clf, self.config.saveModel)
        return val_xgb_pre, test_xgb_pre, score_xgb

    def train(self, X_train, X_valid, labels_train, labels_valid, X_test, param, retrain=False, save_type='J'):
        if self.modelType == 'LGB':
            val_lgb_pre, test_lgb_pre, score_lgb = self.LGB_train(X_train, X_valid, labels_train, labels_valid,
                                                                  X_test, param, retrain)
            return val_lgb_pre, test_lgb_pre, score_lgb
        elif self.modelType == 'XGB':
            val_xgb_pre, test_xgb_pre, score_xgb = self.XGB_train(X_train, X_valid, labels_train, labels_valid,
                                                                  X_test, param, retrain)
            return val_xgb_pre, test_xgb_pre, score_xgb
        else:
            print('请确认模型名称')
            return None

