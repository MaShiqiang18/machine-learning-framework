# coding=gbk

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold, StratifiedKFold
from trainModels.untils.defindMetrics import MyMetrics

def stackModel(train_stack, test_stack, y_train, myMetrics_type, n_runs=3, n_folds=5, use_StratifiedKFold=True):

    predictions_stack_sum_of_runs = np.zeros(len(test_stack))
    oof_stack_sum_of_runs = np.zeros(len(train_stack))

    for run in range(n_runs):
        predictions = np.zeros(test_stack.shape[0])
        oof_stack_pre = np.zeros(train_stack.shape[0])
        random_seed = 2015 + 1000 * (run + 1)
        if use_StratifiedKFold:
            sfolder = StratifiedKFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        else:
            sfolder = KFold(n_splits=n_folds, random_state=random_seed, shuffle=True)
        skf_tmp = sfolder.split(train_stack, y_train)

        for fold_, (trn_idx, val_idx) in enumerate(skf_tmp):
            print("Stack Run: %d, Fold: %d" % (run+1, fold_+1))

            trn_data, trn_y = train_stack[trn_idx], y_train.iloc[trn_idx]
            val_data, val_y = train_stack[val_idx], y_train.iloc[val_idx]

            clf_3 = BayesianRidge()
            clf_3.fit(trn_data, trn_y)

            oof_stack_pre[val_idx] = clf_3.predict(val_data)
            predictions += clf_3.predict(test_stack) / n_folds

        score = MyMetrics(myMetrics_type).metricsFunc(oof_stack_pre, y_train)
        print("Stack Run: {}, CV val score: {:<8.5f}".format(run + 1, score))

        predictions_stack_sum_of_runs = predictions_stack_sum_of_runs + predictions
        oof_stack_sum_of_runs = oof_stack_sum_of_runs + oof_stack_pre

    predictions_stack_mean_of_runs = predictions_stack_sum_of_runs / n_runs
    oof_stack_mean_of_runs = oof_stack_sum_of_runs / n_runs

    finalScore = MyMetrics(myMetrics_type).metricsFunc(oof_stack_mean_of_runs, y_train)
    print("Final score: {}".format(finalScore))
    return predictions_stack_mean_of_runs, oof_stack_mean_of_runs, finalScore


# train_stack = np.vstack([oof_lgb, oof_xgb]).transpose()
# test_stack = np.vstack([predictions_lgb, predictions_xgb]).transpose()