# coding=gbk

from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
存放不同的神经网络结构：FC_Resreg，FC_8Dmodel，FC_6Dmodel
"""

class FC_Resreg(object):
    def res_block(self, X, stage, d0, d1, d2, d3, block):
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        X_shortcut = X

        X = keras.layers.Dense(input_dim=d0, units=d1, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.LeakyReLU(0.3)(X)
        X = keras.layers.BatchNormalization(axis=-1, name=bn_name_base + '2a')(X)
        X = keras.layers.Dropout(rate=0.3)(X)

        X = keras.layers.Dense(input_dim=d1, units=d2, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.LeakyReLU(0.3)(X)
        X = keras.layers.BatchNormalization(axis=-1, name=bn_name_base + '2b')(X)
        X = keras.layers.Dropout(rate=0.3)(X)

        X = keras.layers.Dense(input_dim=d2, units=d3, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)
        X = keras.layers.LeakyReLU(0.3)(X)
        X = keras.layers.BatchNormalization(axis=-1, name=bn_name_base + '2c')(X)
        X = keras.layers.Dropout(rate=0.3)(X)

        X = keras.layers.Add()([X, X_shortcut])
        X = keras.layers.LeakyReLU(0.3)(X)
        return X

    def structure(self, input_shape=(30,)):
        X_input = keras.layers.Input(input_shape)

        X = keras.layers.Dense(units=36)(X_input)
        X = keras.layers.LeakyReLU(0.3)(X)
        X = keras.layers.BatchNormalization(axis=-1, name='bn1')(X)
        X_inp1 = X

        X = self.res_block(X, stage=2, d0=36, d1=46, d2=63, d3=36, block='b')

        X = self.res_block(X, stage=3, d0=36, d1=66, d2=76, d3=36, block='b')

        X = self.res_block(X, stage=4, d0=36, d1=76, d2=86, d3=36, block='b')

        X = self.res_block(X, stage=5, d0=36, d1=86, d2=96, d3=36, block='b')

        X = self.res_block(X, stage=6, d0=36, d1=96, d2=106, d3=36, block='b')

        X = self.res_block(X, stage=7, d0=36, d1=106, d2=116, d3=36, block='b')

        X = self.res_block(X, stage=8, d0=36, d1=116, d2=126, d3=36, block='b')

        X = keras.layers.Add()([X, X_inp1])

        X = keras.layers.Dense(units=1, kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                               activation='relu')(X)

        X = keras.layers.Dense(units=10, kernel_initializer=keras.initializers.glorot_uniform(seed=0),
                               activation='relu')(X)
        X = keras.layers.BatchNormalization(axis=-1, name='bn2')(X)

        X = keras.layers.Dense(units=1, kernel_initializer=keras.initializers.glorot_uniform(seed=0))(X)

        model = keras.Model(X_input, X)

        return model


class FC_8Dmodel(object):
    def structure(self, input_dim):
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(512, input_dim=input_dim, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(0.02)))
        model.add(keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
        model.add(keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))

        model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
        model.add(keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
        model.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
        model.add(keras.layers.Dense(8, activation='relu', kernel_regularizer=keras.regularizers.l2(0.02)))
        model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.l2(0.02)))
        return model


class FC_6Dmodel(object):
    def structure(self, input_dim):
        init = keras.initializers.glorot_uniform(seed=1)
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(units=300, input_dim=input_dim, kernel_initializer=init, activation='softplus'))
        # model.add(Dropout(0.2))
        model.add(keras.layers.Dense(units=300, kernel_initializer=init, activation='softplus'))
        # model.add(Dropout(0.2))
        model.add(keras.layers.Dense(units=64, kernel_initializer=init, activation='softplus'))
        model.add(keras.layers.Dense(units=32, kernel_initializer=init, activation='softplus'))
        model.add(keras.layers.Dense(units=8, kernel_initializer=init, activation='softplus'))
        model.add(keras.layers.Dense(units=1))
        return model




