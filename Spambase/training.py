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


import sys
sys.path.append("../")
import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from SSA import SSA
from ModelTest import ModelTest


DATA_PATH = 'spambase.data'

def load_data(n_folds):
    np.random.seed(314159)
    import pandas as pd
    data = pd.DataFrame.from_csv(
        path=DATA_PATH, header=None, index_col=None, sep=",")
    data = data.sample(frac=1).dropna(axis=0)
    data = pd.get_dummies(data).as_matrix()
    X, y = data[:, :-1].astype(np.float32), data[:, -1].astype(np.int32)
    y = np.hstack((y[:, None], 1-y[:, None]))
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
        y_train = np.vstack([folds[fold][1] for fold in train_folds])
        X_valid, y_valid = folds[valid_fold]
        X_test, y_test = folds[test_fold]
        dataset.append([X_train, y_train, X_valid, y_valid, X_test, y_test])
        print(X_valid.shape, y_valid.shape, X_test.shape, y_test.shape)
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
model.add(Dense(32, input_dim=57, kernel_initializer='uniform'))
# model.add(Dropout(0.85))
model.add(SSA(32, input_dim=32))
model.add(Dense(2, input_dim=32,  kernel_initializer='uniform'))
import keras.backend as K

def mse_sample_mean(Y_true, Y_pred):
    return Y_pred

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

modeltest_1 = ModelTest(X_train,
                        mean_y_train + std_y_train * np.atleast_2d(Y_train), 
                        test_every_X_epochs=1, verbose=0, loss='categorical', 
                        mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=batch_size)
modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test), 
                        test_every_X_epochs=1, verbose=0, loss='categorical', 
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


