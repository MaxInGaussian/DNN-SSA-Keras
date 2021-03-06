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


import sys, os
import numpy as np
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from keras import backend as K
from keras import models
from six.moves import xrange


class StochasticTrainer(Callback):
    
    ''' Test model at the end of every X epochs.
    The model is tested using both MC dropout and the dropout
    approximation. Output metrics for various losses are supported.
    # Arguments
        task: a string from ['regression', 'classification']
            used to distinguish the testing metric.
        datasets: a list of datasets, i.e. list of [X, Y],
            from which we get accuracy / error (ground truth) along training.
        n_samples: number of stochastic passes to obtain empirical distribution.
        valid_freq: test every valid_freq epochs.
        batch_size: number of data points to put in each batch
            (often larger than training batch size).
        mean_y_train: mean of outputs in regression cases to add back
            to model output ('regression' task).
        std_y_train: std of outputs in regression cases to add back
            to model output ('regression' task).
        verbose: verbosity mode, True or False.
    # References
    '''
    def __init__(self, task, datasets, valid_freq, n_samples, batch_size,
        min_delta, patience, evaluation, save_path=None, verbose=False, **args):
        super(StochasticTrainer, self).__init__()
        self.task = task
        self.datasets = datasets
        self.n_samples = n_samples
        self.valid_freq = valid_freq
        self.batch_size = batch_size
        self.verbose = verbose
        self.evaluation = evaluation
        self.save_path = save_path
        if(save_path is not None):
            save_path_dir = ''.join(save_path.split('/')[:-1])
            if(len(save_path_dir)):
                if not os.path.exists(save_path_dir):
                    os.makedirs(save_path_dir)
        self.min_delta = min_delta
        self.patience = patience
        self.reused_best = False
        self.wait = 0
        self.stopped_epoch = 0
        self.monitor_op = np.less
        self._predict_stochastic = None

    def predict_stochastic(self, X, batch_size=128, verbose=False):
        '''Generate output predictions for the input samples batch by batch,
        using stochastic forward passes. This procedure can be used for SGPA.
        # Arguments
            X: the input data, as a numpy array.
            batch_size: integer.
            verbose: verbosity mode, True or False.
        # Returns
            A numpy array of predictions.
        '''
        def standardize_X(X):
            if type(X) == list:
                return X
            else:
                return [X]
        X = standardize_X(X)
        if self._predict_stochastic is None:
            self._predict_stochastic = K.Function(
                self.model.inputs+[K.learning_phase()], self.model.outputs)
        return self.model._predict_loop(
            self._predict_stochastic, X, batch_size, verbose)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        if(not self.evaluation and self.save_path is not None):
            if os.path.exists(self.save_path):
                self.model.load_weights(self.save_path)
    
    def on_train_end(self, logs=None):
        if(self.save_path is not None):
            self.model.load_weights(self.save_path)
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.valid_freq != 0:
            return
        for data_id, (X, Y) in enumerate(self.datasets):
            noise_var = np.exp(self.model.output_layers[0].get_weights()[0])
            Y_preds = np.array([self.predict_stochastic(
                X, batch_size=self.batch_size, verbose=self.verbose)
                    for _ in range(self.n_samples)])
            Y_preds_mean = np.mean(Y_preds, 0)
            Y_preds_var = np.var(Y_preds, 0)+noise_var
            if self.task == 'regression':
                rmse = np.sqrt(np.mean((Y-Y_preds_mean)**2))
                nlpd = .5*(np.mean(np.log(Y_preds_var)+((
                    Y-Y_preds_mean)**2)/Y_preds_var)+np.log(2*np.pi))
                if(data_id == 0):
                    self.current = rmse*nlpd
                print("Epoch %05d: RMSE of (X_valid_%d, Y_valid_%d) = %0.5f %d"%(
                    epoch, data_id, data_id, float(rmse), self.wait))
                print("Epoch %05d: NLPD of (X_valid_%d, Y_valid_%d) = %0.5f %d"%(
                    epoch, data_id, data_id, float(nlpd), self.wait))
            elif self.task == 'classification':
                acc = np.mean(np.argmax(Y, -1)==np.argmax(Y_preds_mean, -1))
                print("MC accuracy at epoch %05d = %0.5f" % (epoch, float(acc)))
                if(data_id == 0):
                    self.current = (1-acc)
            else:
                raise Exception('No task: '+self.task)
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.valid_freq != 0:
            return
        logs = logs or {}
        if self.monitor_op(self.current+self.min_delta, self.best):
            self.wait = 0
            self.reused_best = True
            self.best = self.current
            if(self.save_path is not None):
                self.model.save_weights(self.save_path, overwrite=True)
        else:
            self.wait += 1
            if(self.wait >= self.patience):
                self.stopped_epoch = epoch
                self.model.stop_training = True