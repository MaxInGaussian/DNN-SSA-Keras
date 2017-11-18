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
from keras import initializers
from keras.engine import InputSpec
from keras.engine.topology import Layer


class SSA(Layer):
    """This layer implements the stochastic spectral acivations (SSA) technique.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(SSA(Dense(8), input_shape=(16), ))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(SSA(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `SSA` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:
    ```python
        model = Sequential()
        model.add(SSA(Conv2D(64, (3, 3)), input_shape=(299, 299, 3)))
    ```
    # Arguments
        output_dim: a positive number which defines the dimension of outputs.
        input_dim: a positive number which defines the dimension of inputs.
        use_mc_dropout: a boolean that defines whether using MC-Dropout or not.
    """

    def __init__(self, output_dim, input_dim, **kwargs):
        assert output_dim%2==0, "output_dim must be an even integer for SSA!"
        super(SSA, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.input_dim = input_dim

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        self.alpha_mean = K.random_normal_variable(
            mean=0.,
            scale=1.,
            shape=(self.input_dim, self.output_dim//2),
        )
        self.alpha_logstd = K.random_normal_variable(
            mean=0.,
            scale=1.,
            shape=(self.input_dim, self.output_dim//2),
        )
        super(SSA, self).build(input_shape)

    def call(self, Z):
        epsilon1 = tf.random_normal((self.input_dim, self.output_dim//2))
        A1 = self.alpha_mean+K.exp(self.alpha_logstd)*epsilon1
        epsilon2 = tf.random_normal((self.input_dim, self.output_dim//2))
        A2 = self.alpha_mean+K.exp(self.alpha_logstd)*epsilon2
        t1 = K.dot(Z, A1)/np.sqrt(self.output_dim//2)
        t2 = K.dot(Z, A2)/np.sqrt(self.output_dim//2)
        t = K.concatenate([K.cos(t1)+K.cos(t2), K.sin(t1)+K.sin(t2)], -1)
        return t/np.sqrt(self.input_dim)
        