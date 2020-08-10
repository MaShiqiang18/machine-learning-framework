# coding=gbk

import pandas as pd
import numpy as np
from trainModels.untils.trainKerasModel import loadKerasModel
from trainModels.untils import getKfoldIndex, defindMetrics
import joblib
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def predict_by_models(X_train, X_test, config):
    """
    --��ʵ��run_train.py�ļ򻯰棺ֻ���á���ѵ��
    ����ѵ��������label�������Լ�������label����
    ��������ѵ����ѭ��runs�Σ�ÿ�β�ͬ������ӣ���
    ÿ��ʹ��folds�۽��棬
    ÿ��ʹ��bagging_size��bagging��ÿ��bagging�����ȡbootstrap_ratio*100%��ѵ�����������Ԥ�������bagging�ľ�ֵ
    ����ÿ��run��folds��ѵ����ѵ������Ԥ�����Ǹ�foldsƴ�ӣ������Լ���Ԥ������folds�εľ�ֵ
    �����ѵ�����Ͳ��Լ���runs�εľ�ֵ
    :param X_train: ѵ����
    :param X_test: ���Լ�
    :return: ѵ�����Ͳ��Լ���Ԥ��ֵ��ѵ�����ĵ÷�
    """
    skf = getKfoldIndex.get_kfold_index(X_train, config)

    target_col = config.data_label
    y_train = X_train.loc[:, target_col]
    X_train = X_train.drop([target_col], axis=1)

    metrics_name = config.metrics_name
    myMetrics = defindMetrics.MyMetrics(metrics_name)

    #######################
    ## Generate Features ##
    #######################

    print("ͬ־��������ߣ�����Ӹ�����������")
    Loss_cv = np.zeros((config.n_runs, config.n_folds), dtype=float)
    pre_sumRuns_test = np.zeros(len(X_test))
    pre_sumRuns_train = np.zeros(len(X_train))

    for run in range(config.n_runs):
        info_0 = "<=============================================================================>"
        print(info_0)

        predictions_run = np.zeros(len(X_test))
        val_oneRun = pd.DataFrame()
        for fold_, (trn_idx, val_idx) in enumerate(skf[run]):

            info_1 = "*******************************************************************************"
            info_2 = "************************* Model Run: %d, Fold: %d start *************************" % (run+1, fold_+1)
            info_3 = "*******************************************************************************\n"

            print(info_1)
            print(info_2)
            print(info_3)

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

            # ʹ��Bagging��ܣ�ѵ������������ѵ��
            for n in range(config.bagging_size):
                info_4 = '---------------------------------->bagging: {}'.format(n)
                print(info_4)

                if config.bootstrap_replacement:
                    # ���ѡ
                    sampleSize = int(numTrain*config.bootstrap_ratio)
                    index_base = rng.randint(numTrain, size=sampleSize)
                    index_meta = [i for i in range(numTrain) if i not in index_base]
                else:
                    # ����ѡ
                    randnum = rng.uniform(size=numTrain)
                    index_base = [i for i in range(numTrain) if randnum[i] < config.bootstrap_ratio]
                    index_meta = [i for i in range(numTrain) if randnum[i] >= config.bootstrap_ratio]

                model_name = 'run_{}_fold_{}_bag_{}'.format(run, fold_, n)


                joblib_models_path = config.path_data.saveModelPath + '/joblib_models/%s/' % config.modelType \
                                     + model_name + '.pkl'
                pickle_models_path = config.path_data.saveModelPath + '/pickle_models/%s/' % config.modelType\
                                     + model_name + '.txt'
                keras_models_path = config.path_data.saveModelPath +'/keras_models/%s/' % config.modelType \
                                     + model_name + ".hdf5"


                if os.path.exists(joblib_models_path):
                    info_5 = '����ģ��Ԥ�⣺{}'.format(model_name + '.pkl')
                    print(info_5)
                    model_load = joblib.load(joblib_models_path)
                elif os.path.exists(pickle_models_path):
                    info_6 = '����ģ��Ԥ�⣺{}'.format(model_name + '.txt')
                    print(info_6)
                    with open(pickle_models_path, 'rb') as f:
                        model_load = pickle.load(f)
                elif os.path.exists(keras_models_path):
                    info_7 = '����ģ��Ԥ�⣺{}'.format(model_name + '.hdf5')
                    print(info_7)
                    kerasModel = loadKerasModel(model_name, config)
                    model_load = kerasModel.get_modelStructure(X_train_kf.shape[1])
                    model_load.load_weights(keras_models_path)
                else:
                    print('ģ�Ϳ��в����ڵ���ģ�ͣ����Ƚ�ģ������')
                    return None
                    #############################################################################################

                pred_valid = model_load.predict(X_valid_kf.values)
                preds_bagging_val[:, n] = pred_valid.reshape(-1)
                pred_meanBagging_val = np.mean(preds_bagging_val[:, :(n + 1)], axis=1)
                score_meanBagging_val = myMetrics.metricsFunc(pred_meanBagging_val, labels_valid_kf)

                pred_test = model_load.predict(X_test.values)
                preds_bagging_test[:, n] = pred_test.reshape(-1)
                pred_meanBagging_test = np.mean(preds_bagging_test[:, :(n + 1)], axis=1)

                if (n + 1) != config.bagging_size:
                    info_10 = " - - - - - - - - - - - - - - - - ->[{}-{}-{}] score:{}  shape:{} x {}".format(
                        run+1, fold_+1, n + 1, np.round(score_meanBagging_val, 6), X_train_kf.shape[0], X_train_kf.shape[1])
                    print(info_10)
                else:
                    info_11 = "- - - - - - - - - - - - - - - - ->[{}-{}-{}] score:{}  shape:{} x {}\n\n\n".format(
                        run+1, fold_+1, n + 1, np.round(score_meanBagging_val, 6), X_train_kf.shape[0], X_train_kf.shape[1])
                    print(info_11)

                if n == (config.bagging_size-1):
                    # ĳһ�۵����
                    Loss_cv[run, fold_] = score_meanBagging_val
                    # ��������֤������
                    val_oneFold_tmp = pd.DataFrame(data=pred_meanBagging_val, index=val_idx)
                    val_oneRun = pd.concat([val_oneRun, val_oneFold_tmp], axis=0)

                    predictions_run += pred_meanBagging_test / config.n_folds

        # һ��run����֤�����
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
        print(info_12)
        info_13 = "<=============================================================================>\n\n\n\n"
        print(info_13)

        pre_sumRuns_test = pre_sumRuns_test + predictions_run
        pre_sumRuns_train = pre_sumRuns_train + val_oneRun.iloc[:, 0].values

    pre_sumRuns_test = pre_sumRuns_test/config.n_runs
    pre_sumRuns_train = pre_sumRuns_train/config.n_runs

    pre_sumRuns_train = pd.DataFrame(pre_sumRuns_train, index=X_train.index)
    score_final = int(myMetrics.metricsFunc(pre_sumRuns_train, y_train))
    
    info_14 = '-------------------------------------------------------------------------------\n' + \
            '-------------------------------------------------------------------------------\n' + \
             '���յ÷�: %d\n\n\n\n\n\n' % score_final
    print(info_14)

    predict_final_dir = config.path_data.predictPath + '/final/%s' % config.modelType
    if not os.path.exists(predict_final_dir):
        os.makedirs(predict_final_dir)

    train_file = predict_final_dir + "/{}_train_{}.csv".format(config.mark, score_final)
    pre_sumRuns_train.to_csv(train_file, index=True)

    test_file = predict_final_dir + "/{}_test_{}.csv".format(config.mark, score_final)
    pre_sumRuns_test = pd.DataFrame(pre_sumRuns_test, index=X_test.index)
    pre_sumRuns_test.to_csv(test_file, index=True)

    return pre_sumRuns_train, pre_sumRuns_test, score_final
