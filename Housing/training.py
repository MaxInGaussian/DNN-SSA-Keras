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
import importlib
import numpy as np
import keras.backend as K
from keras.callbacks import TensorBoard
from Callbacks import StochasticTrainer
from Models import MCDropout_DNN, SGPA_DNN


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
activations = ['tanh', 'sigmoid', 'relu']
layer_sizes = [13, 50, 30, 1]

model = SGPA_DNN(layer_sizes)

X_train, Y_train, X_valid, Y_valid, X_test, Y_test = dataset[0]
X_train, X_valid, X_test, mean_X_train, std_X_train =\
    standardize(X_train, X_valid, X_test)
Y_train, Y_valid, Y_test, mean_y_train, std_y_train =\
    standardize(Y_train, Y_valid, Y_test)
    
echo_datasets = [[X_valid, Y_valid]]
strainer = StochasticTrainer(
    'regression', echo_datasets, valid_freq=10, n_samples=50, save_path='trained/test.hdf5',
    batch_size=batch_size, mean_y_train=mean_y_train, std_y_train=std_y_train,
    min_delta=1e-3, patience=10)

training_setting = {
    'batch_size': batch_size,
    'epochs': 10000,
    'callbacks': [strainer],
}

model.fit(X_train, Y_train, **training_setting)

# approximation for test data:

test_n_samples = 100
prob = np.array([strainer.predict_stochastic(
    X_test, batch_size=100, verbose=0) for _ in range(test_n_samples)])
prob_mean = np.mean(prob, 0)
print(np.sqrt(np.mean(((Y_test-prob_mean)*std_y_train)**2)))


