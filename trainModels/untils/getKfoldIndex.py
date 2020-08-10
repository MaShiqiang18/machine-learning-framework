# encoding=gbk
"""
__file__

    gen_kfold.py

__description__

    This file generates the StratifiedKFold indices which will be kept fixed in
    ALL the following model building parts.
    将重复runs的K折交叉验证的数据的index单独保存
    保存的格式的二维数组，第一维表示第几次run，第二维表示该轮run下的第几折
    例：(runs=3,kfold=3)
    [[([run_0_fold_0_train],[run_0_fold_0_val]),([run_0_fold_1_train],[run_0_fold_1_val]),([run_0_fold_2_train],[run_0_fold_2_val])],
    [([run_1_fold_0_train],[run_1_fold_0_val]),([run_1_fold_1_train],[run_1_fold_1_train]),([run_1_fold_2_train],[run_1_fold_2_train])],
    [([run_2_fold_0_train],[run_2_fold_0_val]),([run_2_fold_1_train],[run_2_fold_1_train]),([run_2_fold_2_train],[run_2_fold_2_train])]]
__author__

    Chenglong Chen < c.chenglong@gmail.com >

"""

import sys
import os
import dill as cPickle
from sklearn.model_selection import KFold, StratifiedKFold
sys.path.append("../")


def get_kfold_index(dfTrain, config, print_out=False):
    skf = []
    skf_00 = []

    data_Y = dfTrain.loc[:, config.data_label]
    data_X = dfTrain

    if config.use_StratifiedKFold:
        Kfold_type = 'stratifiedKFold'
    else:
        Kfold_type = 'simpleKFold'

    file_name = 'runs%d_folds_%d' % (config.n_runs, config.n_folds)
    file_path = config.path_data.dataOriPath
    if not config.index_rfg:
        name_rfg = '[TrainNum_{}]'.format(dfTrain.shape[0])
    else:
        name_rfg = config.index_rfg
    file_Kfold = "/%s_%s_%s.pkl" % (name_rfg, Kfold_type, file_name)
    path_Kfold = file_path + file_Kfold

    if os.path.exists(path_Kfold):
        with open(path_Kfold, "rb") as f:
            skf_out = cPickle.load(f)
            print('使用已有分割好的index：{}'.format(file_Kfold[1:]))
            return skf_out
    else:
        for run in range(config.n_runs):
            random_seed = 2015 + 1000 * (run+1)
            # random_seed = 2018

            if config.use_StratifiedKFold:
                sfolder = StratifiedKFold(n_splits=config.n_folds, random_state=random_seed, shuffle=True)
            else:
                sfolder = KFold(n_splits=config.n_folds, random_state=random_seed, shuffle=True)
            skf_tmp = sfolder.split(data_X, data_Y)
            skf.append(skf_tmp)
            skf_list = []
            for fold, (trainInd, validInd) in enumerate(skf[run]):
                if print_out == True:
                    print("================================")
                    print("Index for run: %s, fold: %s" % (run + 1, fold + 1))
                    print("Train (num = %s)" % len(trainInd))
                    print(trainInd[:10])
                    print("Valid (num = %s)" % len(validInd))
                    print(validInd[:10])
                cell = (trainInd, validInd)
                skf_list.append(cell)
            skf_00.append(skf_list)

        with open(path_Kfold, "wb") as f:
            cPickle.dump(skf_00, f)

        return skf_00