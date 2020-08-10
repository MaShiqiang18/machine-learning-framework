# coding=gbk

import numpy as np
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow_core.python.keras.callbacks import LearningRateScheduler
from models.FCModelLib import FC_Resreg, FC_8Dmodel, FC_6Dmodel
from trainModels.untils.showFigs import metrice_loss_figs
from trainModels.untils import defindMetrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class loadKerasModel(object):
    def __init__(self, modelName, config):
        self.modelName = modelName
        self.config = config
        self.modelType = config.modelType
        if self.modelType not in config.kerasModelLib:
            print('请确认模型类型，暂时不存在这类模型！！！')
        self.savePath = config.path_data.saveModelPath

    def get_modelStructure(self, h):
        # 调用现有模型结构
        if self.config.modelType == 'Resreg':
            model = FC_Resreg().structure(input_shape=(h,))
        elif self.config.modelType == 'fc_6Dmodel':
            model = FC_6Dmodel().structure(h)
        elif self.config.modelType == 'fc_8Dmodel':
            model = FC_8Dmodel().structure(h)
        else:
            print('不存在这类模型：{}'.format(self.config.modelType))
            model = None
        return model

    def train(self, X_train, X_val, y_train, y_val, X_test, params, retrain=False, save_type='J'):

        if 'epochs' in params:
            epochs = params['epochs']
        else:
            epochs = 20
        if 'batch_size' in params:
            batch_size = params['batch_size']
        else:
            batch_size = X_train.shape[0] // 10
        if 'show_fig' in params:
            show_fig = params['show_fig']
        else:
            show_fig = False
        if 'loss' in params:
            loss = params['loss']
        else:
            loss = 'loss'
        if 'optimizer' in params:
            optimizer = params['optimizer']
        else:
            optimizer = 'adam'
        if 'metrics' in params:
            metrics = params['metrics']
            if str(type(metrics)).split('\'')[1] == 'str':
                metrics = [metrics]
        else:
            metrics = 'mse'

        # 利用回调函数，调整训练过程的学习率
        #     def scheduler(epoch):
        #         # 到规定的epoch，学习率减小为原来的1/10
        #
        #         if epoch == 300:
        #             lr = K.get_value(model.optimizer.lr)
        #             K.set_value(model.optimizer.lr, lr * 0.5)
        #             print("lr changed to {}".format(lr * 0.5))
        #         if epoch == 500:
        #             lr = K.get_value(model.optimizer.lr)
        #             K.set_value(model.optimizer.lr, lr * 0.2)
        #             print("lr changed to {}".format(lr * 0.2))
        #         if epoch == 1000:
        #             lr = K.get_value(model.optimizer.lr)
        #             K.set_value(model.optimizer.lr, lr * 0.5)
        #             print("lr changed to {}".format(lr * 0.5))
        #         if epoch == 1500:
        #             lr = K.get_value(model.optimizer.lr)
        #             K.set_value(model.optimizer.lr, lr * 0.2)
        #             print("lr changed to {}".format(lr * 0.2))
        #         if epoch == 2000:
        #             lr = K.get_value(model.optimizer.lr)
        #             K.set_value(model.optimizer.lr, lr * 0.5)
        #             print("lr changed to {}".format(lr * 0.5))
        #         if epoch == 2500:
        #             lr = K.get_value(model.optimizer.lr)
        #             K.set_value(model.optimizer.lr, lr * 0.2)
        #             print("lr changed to {}".format(lr * 0.2))
        #         return K.get_value(model.optimizer.lr)

        def scheduler(epoch):
            # 每隔100个epoch，学习率减小为原来的1/10
            if epoch % 20 == 0 and epoch != 0:
                lr = K.get_value(model.optimizer.lr)
                K.set_value(model.optimizer.lr, lr * 0.6)
                print("lr changed to {}".format(lr * 0.6))
            return K.get_value(model.optimizer.lr)

        reduce_lr = LearningRateScheduler(scheduler)

        # 利用回调函数，保存模型
        load_model_dir = self.config.path_data.saveModelPath+'/keras_models/%s/' % self.config.modelType
        if not os.path.exists(load_model_dir):
            os.makedirs(load_model_dir)
        load_model_file = load_model_dir + self.modelName + ".hdf5"


        save_model_path = load_model_file
        checkpoint = keras.callbacks.ModelCheckpoint(save_model_path, monitor='val_loss',
                                                     verbose=0, save_best_only=True, mode='min', skipping=1,
                                                     save_weights_only=True)

        # 利用回调函数，设置早停止机制
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=200, verbose=0,
                                                       mode='auto', baseline=None, restore_best_weights=False)

        # 确定使用的回调函数
        # callback_lists = [checkpoint, reduce_lr, early_stopping]
        callback_lists = [checkpoint, reduce_lr]

        # 数据由DF转化为array
        train_data_x = np.array(X_train)
        train_data_y = np.array(y_train)

        val_data_x = np.array(X_val)
        val_data_y = np.array(y_val)

        #########################################################
        h = X_train.shape[1]
        # 调用现有模型结构
        if self.config.modelType == 'Resreg':
            model = FC_Resreg().structure(input_shape=(h,))
        elif self.config.modelType == 'fc_6Dmodel':
            model = FC_6Dmodel().structure(h)
        elif self.config.modelType == 'fc_8Dmodel':
            model = FC_8Dmodel().structure(h)
        else:
            print('不存在这类模型：{}'.format(self.config.modelType))
            return None
        #########################################################

        if not retrain:
            # 调用已有模型进行增量训练
            load_model_path = save_model_path
            if os.path.exists(load_model_path):
                try:
                    model.load_weights(load_model_path)
                    # 若成功加载前面保存的参数，输出下列信息
                    print("调用已有模型进行增量训练")
                except Exception as e:
                    print(e)
                    print('模型可能参数不匹配')
            else:
                print("不存在已有模型，重新训练")


        # 定义损失函数、优化器、评价函数
        print(loss, optimizer, metrics)
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # 训练模型
        model.fit(train_data_x, train_data_y, epochs=epochs,
                  batch_size=batch_size, validation_data=(val_data_x, val_data_y),
                  callbacks=callback_lists)

        # 绘制训练过程中的loss、metrics曲线
        lossFig_dir = self.config.path_data.saveFigsPath + '/%s' % self.config.modelType
        if not os.path.exists(lossFig_dir):
            os.makedirs(lossFig_dir)
        lossFig_path = lossFig_dir + '/%s' % self.modelName
        metrice_loss_figs(model, lossFig_path, show_fig=show_fig)

        # 验证集、测试集预测
        val_data_y_pre = model.predict(X_val).reshape((model.predict(X_val).shape[0],))
        test_data_y_pre = model.predict(X_test).reshape((model.predict(X_test).shape[0],))

        # 计算验证集得分
        metrics_name = self.config.metrics_name
        myMetrics = defindMetrics.MyMetrics(metrics_name)
        val_score = myMetrics.metricsFunc(val_data_y_pre, y_val)
        return val_data_y_pre, test_data_y_pre, val_score