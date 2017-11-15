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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from SSA import SSA
from ModelTest import ModelTest


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
    
dataset = load_data(5)
batch_size = 20
X_train, Y_train, X_valid, Y_valid, X_test, Y_test = dataset[0]
X_train, X_valid, X_test = standardize(X_train, X_valid, X_test)[:3]
Y_train, Y_valid, Y_test, mean_y_train, std_y_train = standardize(Y_train, Y_valid, Y_test)

K.set_learning_phase(1)
model = Sequential()
model.add(Dense(50, input_dim=13, kernel_initializer='uniform'))
# model.add(Dropout(0.85))
model.add(SSA(50, input_dim=50))
model.add(Dense(25, input_dim=50, kernel_initializer='uniform'))
model.add(SSA(50, input_dim=25))
model.add(Dense(1, input_dim=50,  kernel_initializer='uniform'))
import keras.backend as K

def mse_sample_mean(Y_true, Y_pred):
    return Y_pred

model.compile(loss='mse', optimizer='adam')

modeltest_1 = ModelTest(X_train,
                        mean_y_train + std_y_train * np.atleast_2d(Y_train), 
                        test_every_X_epochs=1, verbose=0, loss='euclidean', 
                        mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=batch_size)
modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test), 
                        test_every_X_epochs=1, verbose=0, loss='euclidean', 
                        mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=batch_size)
# try:
model.fit(X_train, Y_train,
        batch_size=batch_size, nb_epoch=250, callbacks=[modeltest_1, modeltest_2])
# except:
#     pass

standard_prob = model.predict(X_train, batch_size=500, verbose=1)
print(np.sqrt(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train))
               - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5))


# Dropout approximation for test data:
standard_prob = model.predict(X_test, batch_size=500, verbose=1)
print(np.sqrt(np.mean((np.atleast_2d(Y_test) - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5))

# MC dropout for test data:
T = 50
prob = np.array([modeltest_2.predict_stochastic(X_test, batch_size=500, verbose=0)
                 for _ in range(T)])
prob_mean = np.mean(prob, 0)
print(np.sqrt(np.mean((np.atleast_2d(Y_test) - (mean_y_train + std_y_train * prob_mean))**2, 0)**0.5))


