# coding=gbk

"""
__file__

    config.py

__description__

    This file provides global parameter configurations for the project.

__author__

    ShiQiang Ma

"""

import os
import shutil

############
## Config ##
############
class Path_data(object):
    def __init__(self, mark, cleanAllData, cleanMarkData, train_file, test_file):
        path_up = os.path.abspath('../Data/')
        self.dataOriPath = path_up + r'/train_test_data'
        self.path_train_XY = self.dataOriPath + train_file
        self.path_test_XY = self.dataOriPath + test_file

        self.notesPath = path_up + '/notes'

        self.dataPath = self.notesPath + '/%s' % mark
        self.predictPath = self.dataPath + r'/Predict'
        self.saveModelPath = self.dataPath + r'/Models'
        self.saveFigsPath = self.dataPath + r'/Figs'
        self.logPath = self.dataPath + r'/Log'


        ## create feat folder
        if not os.path.exists(path_up):
            os.makedirs(path_up)
        if not os.path.exists(self.dataOriPath):
            os.makedirs(self.dataOriPath)
        if not os.path.exists(self.notesPath):
            os.makedirs(self.notesPath)
        else:
            if cleanAllData:
                # ���ԭ�����м�¼
                shutil.rmtree(self.notesPath)
                os.makedirs(self.notesPath)

        if not os.path.exists(self.dataPath):
            os.makedirs(self.dataPath)
        else:
            if cleanMarkData:
                # �����ǰ���mark��ԭ�ȵļ�¼
                shutil.rmtree(self.dataPath)
                os.makedirs(self.dataPath)

        if not os.path.exists(self.predictPath):
            os.makedirs(self.predictPath)
        if not os.path.exists(self.logPath):
            os.makedirs(self.logPath)
        if not os.path.exists(self.saveModelPath):
            os.makedirs(self.saveModelPath)
        if not os.path.exists(self.saveFigsPath):
            os.makedirs(self.saveFigsPath)




class ParamConfig:
    def __init__(self, mark, mark_text='', cleanMarkData=False):
        """
        :param mark: �����ݼ��нϴ�Ķ�����ģ���иĶ�ʱ�����ڱ�Ǹ���ѵ���Ĳ���
        :param mark_text: ��mark�Ľ�һ����ϸע��
        :param cleanMarkData: ��ΪTrueʱ���Ὣ֮ǰ��ͬmark��Ԥ��ֵ��ģ�͡���־��ͼƬȫ��ɾ��
                               =================> ���ã�ģ�ͻᱻɾ����Ī�õ��õģ�����
        """
        # data params
        self.project_name = 'used_car'
        self.train_file = '/used_car_80features_train.csv'
        self.test_file = '/used_car_80features_test.csv'
        self.index_rfg = '[used_car_80features]'
        self.data_label = 'price'
        self.n_classes = 4
        self.mark = mark
        self.mark_text = mark_text

        # models params
        self.metrics_name = 'MAE'
        # sklearn��ģ�ͱ��淽ʽ����ѡ��'J'(joblib),'P'(pickle),''(������)
        self.saveModel = 'J'
        self.treeModelLib = ['LGB', 'XGB']
        self.kerasModelLib = ['Resreg', 'fc_6Dmodel', 'fc_8Dmodel']
        self.modelType = 'LGB'
        if self.modelType not in self.treeModelLib + self.kerasModelLib:
            print('����training�µ�config.py��ȷ������ģ�����ͣ�����')

        ## CV params
        self.use_StratifiedKFold = False
        self.n_runs = 1
        self.n_folds = 3
        self.bagging_size = 3
        self.bootstrap_ratio = 0.9
        self.bootstrap_replacement = True

        # path params
        self.cleanAllData = False
        self.cleanMarkData = cleanMarkData
        self.path_data = Path_data(self.mark, self.cleanAllData, self.cleanMarkData, self.train_file, self.test_file)

config = ParamConfig(mark='demo')