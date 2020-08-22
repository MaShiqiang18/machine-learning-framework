# coding=gbk

import pandas as pd
import numpy as np
from trainModels.untils.trainTreeModel import loadTreeModel
from trainModels.untils.trainKerasModel import loadKerasModel
from trainModels.untils import getKfoldIndex, time_tran, defindMetrics
from trainModels.untils.defind_log import DefindLog
import joblib
import pickle
import time
import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_model(X_train, X_test, param, config, use_trainedModel=False):
    """
    输入训练集（带label）、测试集（不带label），
    利用所有训练集循环runs次（每次不同随机种子），
    每次使用folds折交叉，
    每折使用bagging_size次bagging（每次bagging随机抽取bootstrap_ratio*100%的训练集），最后预测结果求各bagging的均值
    对于每次run的folds次训练，训练集的预测结果是各folds拼接，，测试集的预测结果是folds次的均值
    最后求训练集和测试集的runs次的均值
    :param X_train: 训练集
    :param X_test: 测试集
    :param param: 类的参数(超参)，可以用于调参
    :param use_trainedModel: （默认）False：表示训练，True：表示测试（当不存在可调用模型时，会临时重新训练）
    :return: 训练集和测试集的预测值，训练集的得分
    """

    mark = config.mark
    # 提供将训练集的拆分为runs次folds折的index
    skf = getKfoldIndex.get_kfold_index(X_train, config)

    target_col = config.data_label
    y_train = X_train.loc[:, target_col]
    X_train = X_train.drop([target_col], axis=1)

    metrics_name = config.metrics_name
    myMetrics = defindMetrics.MyMetrics(metrics_name)

    log_path = config.path_data.logPath + '/%s' % config.modelType
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = '/%s.log' % config.project_name
    log_file = log_path + log_name
    Mylog = DefindLog(log_file).get_logger()

    if len(mark) % 2 == 0:
        long = 70
    else:
        long = 71

    start_time = time.strftime("%Y_%m_%d %H:%M:%S")
    mark_withTime = mark + ' ' + start_time

    null = 33
    single = (long-len(mark_withTime)-2)//2
    u = '#'*long + '\n'
    m = ' '*null + '#'*single + ' ' + mark_withTime + ' ' + '#'*single + '\n'
    d = ' '*null + '#'*long + '\n'
    train_star = u+m+d
    Mylog.info(train_star)
    if config.mark_text != '':
        cu = '+'*35 + '\n'
        cd = ' '*null + '+'*35 + '\n'
        content = cu + ' '*null + config.mark_text + '\n' + cd
        Mylog.info(content)


    #######################
    ## Generate Features ##
    #######################

    print("同志们让我们撸起袖子干起来！！！")
    Loss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    pre_sumRuns_test = np.zeros(len(X_test))
    pre_sumRuns_train = np.zeros(len(X_train))

    # CPU的执行时间,返回的单位是秒
    start_CPU = time.clock()
    # 程序执行时间(=cpu时间 + io时间 + 休眠或者等待时间)
    start_RUN = datetime.datetime.now()


    for run in range(config.n_runs):
        info_0 = "<=============================================================================>"
        Mylog.info(info_0)
        start_CPU_oneRun = time.clock()
        start_RUN_oneRun = datetime.datetime.now()

        predictions_run = np.zeros(len(X_test))
        val_oneRun = pd.DataFrame()
        for fold_, (trn_idx, val_idx) in enumerate(skf[run]):

            info_1 = "*******************************************************************************"
            info_2 = "************************* Model Run: %d, Fold: %d start *************************" % (run+1, fold_+1)
            info_3 = "*******************************************************************************\n"

            Mylog.info(info_1)
            Mylog.info(info_2)
            Mylog.info(info_3)

            start_CPU_oneFold = time.clock()
            start_RUN_oneFold = datetime.datetime.now()

            X_valid_kf = X_train.iloc[val_idx, :]
            labels_valid_kf = y_train.iloc[val_idx]
            X_train_kf = X_train.iloc[trn_idx, :]
            labels_train_kf = y_train.iloc[trn_idx]

            rng = np.random.RandomState(2015 + 1000 * run + 10 * fold_)

            numValid = X_valid_kf.shape[0]
            numTrain = X_train_kf.shape[0]
            numTest = X_test.shape[0]

            preds_bagging_val = np.zeros((numValid, config.bagging_size), dtype=float)
            preds_bagging_test = np.zeros((numTest, config.bagging_size), dtype=float)

            # 使用Bagging框架，训练集采样进行训练
            for n in range(config.bagging_size):
                info_4 = '---------------------------------->bagging: {}'.format(n)
                Mylog.info(info_4)
                
                start_CPU_oneBagging = time.clock()
                start_RUN_oneBagging = datetime.datetime.now()

                if config.bootstrap_replacement:
                    # 随机选
                    sampleSize = int(numTrain*config.bootstrap_ratio)
                    index_base = rng.randint(numTrain, size=sampleSize)
                    index_meta = [i for i in range(numTrain) if i not in index_base]
                else:
                    # 均匀选
                    randnum = rng.uniform(size=numTrain)
                    index_base = [i for i in range(numTrain) if randnum[i] < config.bootstrap_ratio]
                    index_meta = [i for i in range(numTrain) if randnum[i] >= config.bootstrap_ratio]

                model_name = 'run_{}_fold_{}_bag_{}'.format(run+1, fold_+1, n+1)

                if use_trainedModel:
                    joblib_models_path = config.path_data.saveModelPath + '/joblib_models/%s/' % config.modelType\
                                         + model_name + '.pkl'
                    pickle_models_path = config.path_data.saveModelPath + '/pickle_models/%s/' % config.modelType \
                                         + model_name + '.txt'
                    keras_models_path = config.path_data.saveModelPath +'/keras_models/%s/' % config.modelType \
                                         + model_name + ".hdf5"


                    if os.path.exists(joblib_models_path):
                        info_5 = '调用模型预测：{}'.format(model_name + '.pkl')
                        Mylog.info(info_5)
                        model_load = joblib.load(joblib_models_path)
                        pred_valid = model_load.predict(X_valid_kf.values)
                        pred_test = model_load.predict(X_test.values)

                    elif os.path.exists(pickle_models_path):
                        info_6 = '调用模型预测：{}'.format(model_name + '.txt')
                        Mylog.info(info_6)
                        with open(pickle_models_path, 'rb') as f:
                            model_load = pickle.load(f)
                        pred_valid = model_load.predict(X_valid_kf.values)
                        pred_test = model_load.predict(X_test.values)
                    elif os.path.exists(keras_models_path):
                        info_7 = '调用模型预测：{}'.format(model_name + '.hdf5')
                        Mylog.info(info_7)
                        kerasModel = loadKerasModel(model_name, config)
                        model_load = kerasModel.get_modelStructure(X_train_kf.shape[1])
                        model_load.load_weights(keras_models_path)
                        pred_valid = model_load.predict(X_valid_kf.values)
                        pred_test = model_load.predict(X_test.values)
                    else:
                        info_8 = '不存在模型：{}，临时训练'.format(model_name)
                        Mylog.info(info_8)
                        #############################################################################################
                        if config.modelType in config.treeModelLib:
                            model = loadTreeModel(model_name, config)
                        elif config.modelType in config.kerasModelLib:
                            model = loadKerasModel(model_name, config)
                        else:
                            print('请在training下的config.py中确定合适模型类型！！！')
                            return None
                        pred_valid, pred_test, score_bag_one = model.train(X_train_kf.iloc[index_base, :], X_valid_kf,
                                                                           labels_train_kf.iloc[index_base],
                                                                           labels_valid_kf,
                                                                           X_test, param)
                        #############################################################################################
                else:
                    info_9 = '开始训练模型：{}'.format(model_name)
                    Mylog.info(info_9)
                    #############################################################################################
                    if config.modelType in config.treeModelLib:
                        model = loadTreeModel(model_name, config)
                    elif config.modelType in config.kerasModelLib:
                        model = loadKerasModel(model_name, config)
                    else:
                        print('请在training下的config.py中确定合适模型类型！！！')
                        return None
                    pred_valid, pred_test, score_bag_one = model.train(X_train_kf.iloc[index_base, :], X_valid_kf,
                                                                     labels_train_kf.iloc[index_base], labels_valid_kf,
                                                                     X_test, param)
                    #############################################################################################

                if not use_trainedModel:
                    end_CPU_oneBagging = time.clock()
                    end_RUN_oneBagging = datetime.datetime.now()

                    oneBagging_spend_CPU = time_tran.time_format(int(end_CPU_oneBagging - start_CPU_oneBagging))
                    oneBagging_spend_RUN = time_tran.time_format(int((end_RUN_oneBagging - start_RUN_oneBagging).total_seconds()))

                    sumBagging_spend_CPU = time_tran.time_format(int(end_CPU_oneBagging - start_CPU))
                    sumBagging_spend_RUN = time_tran.time_format(int((end_RUN_oneBagging - start_RUN).total_seconds()))
                    Mylog.info('此轮bagging，CPU耗时：{}'.format(oneBagging_spend_CPU))
                    Mylog.info('此轮bagging，RUN耗时：{}'.format(oneBagging_spend_RUN))
                    Mylog.info('从开始运行到现在，CPU耗时：{}'.format(sumBagging_spend_CPU))
                    Mylog.info('从开始运行到现在，RUN耗时：{}'.format(sumBagging_spend_RUN))


                ## this bagging iteration
                preds_bagging_val[:, n] = pred_valid.reshape(-1)
                pred_meanBagging_val = np.mean(preds_bagging_val[:, :(n + 1)], axis=1)
                score_meanBagging_val = myMetrics.metricsFunc(pred_meanBagging_val, labels_valid_kf)

                preds_bagging_test[:, n] = pred_test.reshape(-1)
                pred_meanBagging_test = np.mean(preds_bagging_test[:, :(n + 1)], axis=1)

                if (n + 1) != config.bagging_size:
                    info_10 = " - - - - - - - - - - - - - - - - ->[{}-{}-{}] score:{}  shape:{} x {}".format(
                        run+1, fold_+1, n + 1, np.round(score_meanBagging_val, 6), X_train_kf.shape[0], X_train_kf.shape[1])
                    Mylog.info(info_10)
                else:
                    info_11 = "- - - - - - - - - - - - - - - - ->[{}-{}-{}] score:{}  shape:{} x {}\n\n\n".format(
                        run+1, fold_+1, n + 1, np.round(score_meanBagging_val, 6), X_train_kf.shape[0], X_train_kf.shape[1])
                    Mylog.info(info_11)

                if n == (config.bagging_size-1):
                    # 某一折的误差
                    Loss_cv[run, fold_] = score_meanBagging_val
                    # 将所有验证集集成
                    val_oneFold_tmp = pd.DataFrame(data=pred_meanBagging_val, index=val_idx)
                    val_oneRun = pd.concat([val_oneRun, val_oneFold_tmp], axis=0)

                    predictions_run += pred_meanBagging_test / config.n_folds

                    if not use_trainedModel:
                        end_CPU_oneFold = time.clock()
                        end_RUN_oneFold = datetime.datetime.now()

                        oneFold_spend_CPU = time_tran.time_format(int(end_CPU_oneFold - start_CPU_oneFold))
                        oneFold_spend_RUN = time_tran.time_format(int((end_RUN_oneFold - start_RUN_oneFold).total_seconds()))

                        sumFold_spend_CPU = time_tran.time_format(int(end_CPU_oneFold - start_CPU))
                        sumFold_spend_RUN = time_tran.time_format(int((end_RUN_oneFold - start_RUN).total_seconds()))
                        Mylog.info('此轮fold，CPU耗时：{}'.format(oneFold_spend_CPU))
                        Mylog.info('此轮fold，RUN耗时：{}'.format(oneFold_spend_RUN))
                        Mylog.info('从开始运行到现在，CPU耗时：{}'.format(sumFold_spend_CPU))
                        Mylog.info('从开始运行到现在，RUN耗时：{}'.format(sumFold_spend_RUN))


        # 一次run的验证集结果
        val_oneRun.sort_index(axis=0, ascending=True, inplace=True)
        val_oneRun.index = y_train.index
        this_run_score = myMetrics.metricsFunc(val_oneRun[0], y_train)

        predict_run_dir = config.path_data.predictPath + '/process/run_%d/%s' % (run+1, config.modelType)
        if not os.path.exists(predict_run_dir):
            os.makedirs(predict_run_dir)

        train_run_file = predict_run_dir + "/{}_train_{}.csv".format(config.mark, int(this_run_score))
        val_oneRun_df = pd.DataFrame(val_oneRun, index=y_train.index)
        val_oneRun_df.to_csv(train_run_file, index=True)

        test_run_file = predict_run_dir + "/{}_test_{}.csv".format(config.mark, int(this_run_score))
        test_oneRun_df = pd.DataFrame(predictions_run, index=X_test.index)
        test_oneRun_df.to_csv(test_run_file, index=True)






        info_12 = "******************** Model Run: {}, CV val score: {:<8.5f} ********************".format(
            run+1, this_run_score)
        Mylog.info(info_12)
        info_13 = "<=============================================================================>\n\n\n\n"
        Mylog.info(info_13)

        pre_sumRuns_test = pre_sumRuns_test + predictions_run
        pre_sumRuns_train = pre_sumRuns_train + val_oneRun.iloc[:, 0].values

        if not use_trainedModel:
            end_CPU_oneRun = time.clock()
            end_RUN_oneRun = datetime.datetime.now()

            onoRun_spend_CPU = time_tran.time_format(int(end_CPU_oneRun - start_CPU_oneRun))
            onoRun_spend_RUN = time_tran.time_format(int((end_RUN_oneRun - start_RUN_oneRun).total_seconds()))

            sumRun_spend_CPU = time_tran.time_format(int(end_CPU_oneRun - start_CPU))
            sumRun_spend_RUN = time_tran.time_format(int((end_RUN_oneRun - start_RUN).total_seconds()))
            Mylog.info('此轮run，CPU耗时：{}'.format(onoRun_spend_CPU))
            Mylog.info('此轮run，RUN耗时：{}'.format(onoRun_spend_RUN))
            Mylog.info('从开始运行到现在，CPU耗时：{}'.format(sumRun_spend_CPU))
            Mylog.info('从开始运行到现在，RUN耗时：{}'.format(sumRun_spend_RUN))


    pre_sumRuns_test = pre_sumRuns_test/config.n_runs
    pre_sumRuns_train = pre_sumRuns_train/config.n_runs

    pre_sumRuns_train = pd.DataFrame(pre_sumRuns_train, index=X_train.index)
    score_final = myMetrics.metricsFunc(pre_sumRuns_train, y_train)
    
    info_14 = '-------------------------------------------------------------------------------\n' + \
            ' '*null + '-------------------------------------------------------------------------------\n' + \
            ' '*null + '最终得分: %f\n\n\n\n\n\n' % score_final
    Mylog.info(info_14)

    predict_final_dir = config.path_data.predictPath + '/final/%s' % config.modelType
    if not os.path.exists(predict_final_dir):
        os.makedirs(predict_final_dir)

    train_file = predict_final_dir + "/{}_train_{:<8.4f}.csv".format(config.mark, score_final)
    pre_sumRuns_train.to_csv(train_file, index=True)
    train_file_cp = config.path_data.dataPrePath + "/{}_train_{:<8.4f}.csv".format(config.mark, score_final)
    pre_sumRuns_train.to_csv(train_file_cp, index=True)

    test_file = predict_final_dir + "/{}_test_{:<8.4f}.csv".format(config.mark, score_final)
    pre_sumRuns_test = pd.DataFrame(pre_sumRuns_test, index=X_test.index)
    pre_sumRuns_test.to_csv(test_file, index=True)
    test_file_cp = config.path_data.dataPrePath + "/{}_test_{:<8.4f}.csv".format(config.mark, score_final)
    pre_sumRuns_test.to_csv(test_file_cp, index=True)

    return pre_sumRuns_train, pre_sumRuns_test, score_final




# XY_train = pd.read_csv(config.path_data.path_train_XY, index_col=0).iloc[:200, :]
# X_test = pd.read_csv(config.path_data.path_test_XY, index_col=0).iloc[:50, :]
# # param = {'num_leaves': 30,
# #          'min_data_in_leaf': 30,
# #          'objective': 'regression',
# #          'max_depth': -1,
# #          'learning_rate': 0.01,
# #          "min_child_samples": 30,
# #          "boosting": "gbdt",
# #          "feature_fraction": 0.9,
# #          "bagging_freq": 1,
# #          "bagging_fraction": 0.9,
# #          "bagging_seed": 11,
# #          "metric": 'mae',
# #          "lambda_l1": 0.1,
# #          "verbosity": -1}
#
# param = {
#     'epochs': 20,
#     'batch_size': 12000,
#     'show_fig': False,
#     'loss': 'mean_absolute_error',
#     'optimizer': 'adam',
#     'metrics': 'mae'
# }
#
# pre_sumRuns_train, pre_sumRuns_test, score_final = train_model(XY_train, X_test, param)
# print('score_final')
