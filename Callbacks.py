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
from keras.callbacks import Callback
from keras import backend as K
from keras import models


class StochasticTrainer(Callback):
    
    ''' Test model at the end of every X epochs.
    The model is tested using both MC dropout and the dropout
    approximation. Output metrics for various losses are supported.
    # Arguments
        Xt: model inputs to test.
        Yt: model outputs to get accuracy / error (ground truth).
        T: number of samples to use in MC dropout.
        test_every_X_epochs: test every test_every_X_epochs epochs.
        batch_size: number of data points to put in each batch
            (often larger than training batch size).
        verbose: verbosity mode, 0 or 1.
        loss: a string from ['binary', 'categorical', 'euclidean']
            used to calculate the testing metric.
        mean_y_train: mean of outputs in regression cases to add back
            to model output ('euclidean' loss).
        std_y_train: std of outputs in regression cases to add back
            to model output ('euclidean' loss).
    # References
        - [Dropout: A simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html)
        - [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://arxiv.org/abs/1506.02142)
    '''
    def __init__(self, datasets, valid_freq=1, n_samples=10, batch_size=500,
        verbose=True, loss=None, mean_y_train=None, std_y_train=None):
        super(ModelTest, self).__init__()
        self.Xt = Xt
        self.Yt = np.array(Yt)
        self.T = T
        self.test_every_X_epochs = test_every_X_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss = loss
        self.mean_y_train = mean_y_train
        self.std_y_train = std_y_train
        self._predict_stochastic = None

    def predict_stochastic(self, X, batch_size=128, verbose=0):
        '''Generate output predictions for the input samples
        batch by batch, using stochastic forward passes. If
        dropout is used at training, during prediction network
        units will be dropped at random as well. This procedure
        can be used for MC dropout (see [ModelTest callbacks](callbacks.md)).
        # Arguments
            X: the input data, as a numpy array.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.
        # Returns
            A numpy array of predictions.
        # References
            - [Dropout: A simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html)
            - [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://arxiv.org/abs/1506.02142)
        '''
        def standardize_X(X):
            if type(X) == list:
                return X
            else:
                return [X]
        X = standardize_X(X)
        if self._predict_stochastic is None: # we only get self.model after init
            self._predict_stochastic = K.Function(self.model.inputs + [K.learning_phase()], self.model.outputs)
        return self.model._predict_loop(self._predict_stochastic, X, batch_size, verbose)[0]


    def on_epoch_begin(self, epoch, logs={}):
        if epoch % self.test_every_X_epochs != 0:
            return
        model_output = self.model.predict(self.Xt, batch_size=self.batch_size,
                                          verbose=self.verbose)
        MC_model_output = []
        for _ in range(self.T):
            MC_model_output += [self.predict_stochastic(self.Xt,
                                                   batch_size=self.batch_size,
                                                   verbose=self.verbose)]
        MC_model_output = np.array(MC_model_output)
        MC_model_output_mean = np.mean(MC_model_output, 0)

        if self.loss == 'binary':
            standard_acc = np.mean(self.Yt == np.round(model_output.flatten()))
            MC_acc = np.mean(self.Yt == np.round(MC_model_output_mean.flatten()))
            print("Standard accuracy at epoch %05d: %0.5f" % (epoch, float(standard_acc)))
            print("MC accuracy at epoch %05d: %0.5f" % (epoch, float(MC_acc)))
        elif self.loss == 'categorical':
            standard_acc = np.mean(np.argmax(self.Yt, axis=-1) == np.argmax(model_output, axis=-1))
            MC_acc = np.mean(np.argmax(self.Yt, axis=-1) == np.argmax(MC_model_output_mean, axis=-1))
            print("Standard accuracy at epoch %05d: %0.5f" % (epoch, float(standard_acc)))
            print("MC accuracy at epoch %05d: %0.5f" % (epoch, float(MC_acc)))
        elif self.loss == 'euclidean':
            model_output = model_output * self.std_y_train + self.mean_y_train
            standard_err = np.mean((self.Yt - model_output)**2.0)**0.5
            MC_model_output_mean = MC_model_output_mean * self.std_y_train + self.mean_y_train
            MC_err = np.mean((self.Yt - MC_model_output_mean)**2.0)**0.5
            # if(MC_err < 10):
            #     raise Exception('I know Python!')
            print("Standard error at epoch %05d: %0.5f" % (epoch, float(standard_err)))
            print("MC error at epoch %05d: %0.5f" % (epoch, float(MC_err)))
        else:
            raise Exception('No loss: ' + loss)