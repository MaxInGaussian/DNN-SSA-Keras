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
import tensorflow as tf
import keras.backend as K
from keras import initializers, regularizers
from keras.engine import InputSpec
from keras.engine.topology import Layer


class SGPA(Layer):
    """This layer implements the spectral Gaussian process acivations (SGPA).
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(SGPA(Dense(8), input_shape=(16), ))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(SSA(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `SGPA` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(SGPA(Conv2D(64, (3, 3)), input_shape=(299, 299, 3)))
    ```
    # Arguments
        units: a positive number which defines the dimension of outputs.
        input_dim: a positive number which defines the dimension of inputs.
    """

    def __init__(self, units, input_dim, **kwargs):
        assert units%2 == 0, 'units must be an even number!'
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(SGPA, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.n_basis = self.units//2
        self.at_output = int(self.n_basis==0)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.alpha_mean = self.add_weight(
            shape=(input_dim, self.n_basis+self.at_output),
            initializer=initializers.normal(),
            name='alpha_mean',
            # regularizer=self.kernel_regularizer,
            # constraint=self.kernel_constraint
        )
        self.alpha_logstd = self.add_weight(
            shape=(input_dim, self.n_basis+self.at_output),
            initializer=initializers.normal(),
            name='alpha_logstd',
            # regularizer=self.kernel_regularizer,
            # constraint=self.kernel_constraint
        )
        super(SGPA, self).build(input_shape)
        self.built = True

    def call(self, Z):
        epsilon1 = tf.random_normal((self.input_dim, self.n_basis))
        A1 = self.alpha_mean+K.exp(self.alpha_logstd)*epsilon1
        epsilon2 = tf.random_normal((self.input_dim, self.n_basis))
        A2 = self.alpha_mean+K.exp(self.alpha_logstd)*epsilon2
        t1 = K.dot(Z, A1)/np.sqrt(self.units//2)
        t2 = K.dot(Z, A2)/np.sqrt(self.units//2)
        if(self.at_output):
            t = K.cos(t1)+K.cos(t2)+K.sin(t1)+K.sin(t2)
        else:
            t = K.concatenate([K.cos(t1)+K.cos(t2), K.sin(t1)+K.sin(t2)], -1)
        return t/np.sqrt(self.input_dim)
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)