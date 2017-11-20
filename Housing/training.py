# Copyright 2017 Max W. Y. Lam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
sys.path.append("../")
import numpy as np
import keras.backend as K
from keras import initializers, regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from Layers import SGPA
from Callbacks import StochasticTrainer


DATA_PATH = 'housing.data'

def load_data(n_folds):
    np.random.seed(314159)
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep="[ ^]+")
    data = data.sample(frac=1).dropna(axis=0).as_matrix().astype(np.float32)
    X, y = data[:, :-1], data[:, -1]
    y = y[:, None]
    n_data = y.shape[0]
    n_partition = n_data//n_folds
    n_train = n_partition*(n_folds-1)
    dataset, folds = [], []
    for i in range(n_folds):
        if(i == n_folds-1):
            fold_inds = np.arange(n_data)[i*n_partition:]
        else:
            fold_inds = np.arange(n_data)[i*n_partition:(i+1)*n_partition]
        folds.append([X[fold_inds], y[fold_inds]])
    for i in range(n_folds):
        valid_fold, test_fold = i, (i+1)%n_folds
        train_folds = np.setdiff1d(np.arange(n_folds), [test_fold, valid_fold])
        X_train = np.vstack([folds[fold][0] for fold in train_folds])
        Y_train = np.vstack([folds[fold][1] for fold in train_folds])
        X_valid, Y_valid = folds[valid_fold]
        X_test, Y_test = folds[test_fold]
        dataset.append([X_train, Y_train, X_valid, Y_valid, X_test, Y_test])
    return dataset

def standardize(data_train, data_valid, data_test):
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    train_standardized = (data_train - mean)/std
    valid_standardized = (data_test - mean)/std
    test_standardized = (data_test - mean)/std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return train_standardized, valid_standardized, test_standardized, mean, std
    
dataset = load_data(10)
batch_size = 50
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = dataset[0]
X_train, X_valid, X_test, mean_X_train, std_X_train =\
    standardize(X_train, X_valid, X_test)
Y_train, Y_valid, Y_test, mean_y_train, std_y_train =\
    standardize(Y_train, Y_valid, Y_test)

import keras.backend as K
K.set_learning_phase(1)

model = Sequential()
model.add(Dense(50, input_dim=13))
# model.add(Activation('relu'))
model.add(SGPA(100, input_dim=50))
# model.add(SGPA(50, input_dim=50))
# model.add(SGPA(50, input_dim=50))
# model.add(Dropout(0.85))
output_layer = Dense(1, input_dim=100)
output_logvar = output_layer.add_weight(
    shape=(),
    initializer=initializers.normal(),
    name='output_logvar'
)
model.add(Dense(1, input_dim=50))

def sgpa_mse(Y_true, Y_pred):
    return .5*(output_logvar+K.exp(-1*output_logvar)*K.mean((Y_true-Y_pred)**2))

model.compile(loss='mse', optimizer='adam')

echo_datasets = [
    [X_train, Y_train],
    [X_valid, Y_valid],
    [X_test, Y_test]
]
strainer = StochasticTrainer(
    'regression', echo_datasets, valid_freq=10, n_samples=50,
    batch_size=batch_size, mean_y_train=mean_y_train, std_y_train=std_y_train)
    
# try:
model.fit(X_train, Y_train,
        batch_size=batch_size, nb_epoch=500, callbacks=[strainer])
# except:
#     pass


# approximation for test data:

test_n_samples = 100
prob = np.array([strainer.predict_stochastic(
    X_test, batch_size=100, verbose=0) for _ in range(test_n_samples)])
prob_mean = np.mean(prob, 0)
prob_var = np.var(prob, 0)
Y_pred = prob_mean*std_y_train+mean_y_train
print(np.sqrt(np.mean(((Y_test-prob_mean)*std_y_train)**2)))
print(np.sqrt(np.mean(((Y_test-model.predict(X_test))*std_y_train)**2)))


