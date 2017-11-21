# MIT License
# 
# Copyright (c) 2017 Max W. Y. Lam
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import keras.backend as K
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from Layers import SGPA


def DNN(layer_sizes, activation='relu'):
    K.set_learning_phase(1)
    model = Sequential()
    for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        model.add(Dense(n_out, input_dim=n_in))
        if(l < len(layer_sizes)-2):
            model.add(Activation(activation))
    model.compile(loss='mse', optimizer='adam')
    return model


def MCDropout_DNN(layer_sizes, activation='relu', dropout_rate=0.5):
    K.set_learning_phase(1)
    model = Sequential()
    for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if(l < len(layer_sizes)-2):
            model.add(Dense(n_out, input_dim=n_in))
            model.add(Activation(activation))
            model.add(Dropout(dropout_rate))
        else:
            output_layer = Dense(n_out, input_dim=n_in)
            noise_logvar = output_layer.add_weight(
                shape=(),
                initializer=initializers.constant(np.log(1e-1)),
                name='noise_logvar'
            )
            model.add(output_layer)
    def sgpa_mse(Y_true, Y_pred):
        return .5*(noise_logvar+K.exp(-noise_logvar)*K.mean((Y_true-Y_pred)**2))
    model.compile(loss=sgpa_mse, optimizer='adam')
    return model


def SGPA_DNN(layer_sizes):
    K.set_learning_phase(1)
    model = Sequential()
    for l, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if(l < len(layer_sizes)-2):
            model.add(Dense(n_out, input_dim=n_in))
            model.add(SGPA(n_out, input_dim=n_out))
        else:
            output_layer = Dense(n_out, input_dim=n_in)
            noise_logvar = output_layer.add_weight(
                shape=(),
                initializer=initializers.constant(np.log(1e-1)),
                name='noise_logvar'
            )
            model.add(output_layer)
    def sgpa_mse(Y_true, Y_pred):
        return .5*(noise_logvar+K.exp(-noise_logvar)*K.mean((Y_true-Y_pred)**2))
    model.compile(loss=sgpa_mse, optimizer='adam')
    return model