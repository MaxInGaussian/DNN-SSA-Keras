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
import Models
from keras.callbacks import TensorBoard
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
    

model_choices = ['SGPA_DNN']
dataset = load_data(5)
test_n_samples = 300
train_dict, fit_dict = {}, {}
train_dict['task'] = 'regression'

## Stochastic Training Parameters
train_dict['evaluation'] = True
train_dict['n_samples'] = 50
train_dict['batch_size'] = fit_dict['batch_size'] = 20

## Early Stopping Parameters
train_dict['min_delta'] = 1e-6
train_dict['patience'] = 10
train_dict['valid_freq'] = 15
fit_dict['epochs'] = 10000

## Network Structure Parameters
activations = ['tanh', 'sigmoid', 'relu']
layer_sizes = [13, 50, 30, 1]

performance_log = {model: [] for model in model_choices}
for model in model_choices:
    
    train_dict['save_path'] = model+'('+','.join(map(str, layer_sizes))+').hdf5'

    for X_train, Y_train, X_valid, Y_valid, X_test, Y_test in dataset:
    
        X_train, X_valid, X_test, mean_X_train, std_X_train =\
            standardize(X_train, X_valid, X_test)
        Y_train, Y_valid, Y_test, mean_Y_train, std_Y_train =\
            standardize(Y_train, Y_valid, Y_test)
        
        train_dict['datasets'] = [[X_valid, Y_valid]]
        strainer = StochasticTrainer(**train_dict)

        keras_model = getattr(Models, model)(layer_sizes)
        keras_model.fit(X_train, Y_train, callbacks=[strainer], **fit_dict)
        
        Y_preds = np.array([strainer.predict_stochastic(
            X_test, verbose=0) for _ in range(test_n_samples)])
        Y_preds_mean = np.mean(Y_preds, 0)
        rmse = np.sqrt(np.mean(((Y_test-Y_preds_mean)*std_Y_train)**2))
        noise_var = np.exp(keras_model.output_layers[0].get_weights()[0])
        Y_preds_var = np.var(Y_preds, 0)+noise_var
        nlpd = .5*(np.mean(np.log(Y_preds_var*std_Y_train**2.)+((
            Y_test-Y_preds_mean)**2)/Y_preds_var)+np.log(2*np.pi))
        print("RMSE of (X_test, Y_test) = %0.5f"%(float(rmse)))
        print("NLPD of (X_test, Y_test) = %0.5f"%(float(nlpd)))
        
        performance_log[model].append([float(rmse), float(nlpd)])

for model in model_choices:
    mu = np.mean(performance_log[model], 0)
    std = np.std(performance_log[model], 0)
    print('>>>> '+model)
    print('     RMSE = {:.4f} \pm {:.4f}'.format(mu[0], 1.96*std[0]))
    print('     NLPD = {:.4f} \pm {:.4f}'.format(mu[1], 1.96*std[1]))