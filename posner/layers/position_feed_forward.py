# -*- coding: utf-8 -*-
"""
.. module:: position_feed_forward
   :synopsis:

.. moduleauthor:: LeeKLTW

"""
from tensorflow import keras
from tensorflow.keras import backend as K

class PositionFeedForward(keras.layers.Layer):

  def __init__(self,
               units,
               activation="relu",
               use_bias=True,
               kernel_initializer="glorot_normal",
               bias_initializer="zeros",
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               dropout_rate=0.0,
               **kwargs):
    super(PositionFeedForward,self).__init__(**kwargs)
    self.units = units
    self.activation = keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.bias_initializer = keras.initializers.get(bias_initializer)
    self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = keras.constraints.get(kernel_constraint)
    self.bias_constraint = keras.constraints.get(bias_constraint)
    self.dropout_rate = dropout_rate
    self.W1, self.W2, self.b1, self.b2 = None, None, None, None

  def get_config(self):
    config = {
       "units":self.units,
       "activation":keras.activations.serialize(self.activation),
       "use_bias":self.use_bias,
       "kernel_initializer":keras.initializers.serialize(self.kernel_initializer),
       "bias_initializer":keras.initializers.serialize(self.bias_initializer),
       "kernel_regularizer":keras.regularizers.serialize(self.kernel_regularizer),
       "bias_regularizer":keras.regularizers.serialize(self.bias_regularizer),
       "kernel_constraint":keras.constraints.serialize(self.kernel_constraint),
       "bias_constraint":keras.constraints.serialize(self.bias_constraint),
       "dropout_rate":self.dropout_rate,
    }
    base_config = super(PositionFeedForward,self).get_config()
    return dict(list(base_config.items())+list(config.items()))

  def compute_output_shape(self, input_shape):
    return input_shape

  def compute_mask(self, inputs, mask=None):
    return mask

  def build(self, input_shape):
    feature_dim = int(input_shape[-1])
    self.W1 = self.add_weight(
      shape=(feature_dim, self.units),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      name=f"{self.name}_W1",
    )
    if self.use_bias:
      self.b1 = self.add_weight(
        shape=(self.units,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        name=f"{self.name}_b1",
      )

    self.W2 = self.add_weight(
      shape=(self.units, feature_dim),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      name=f"{self.name}_W2",
    )

    if self.use_bias:
      self.b2 = self.add_weight(
        shape=(feature_dim,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        name=f"{self.name}_b2",
      )
    super(PositionFeedForward,self).build(input_shape)

  def call(self,x,mask=None,training=None):
    h = K.dot(x, self.W1)
    if self.use_bias:
      h = K.bias_add(h, self.b1)
    if self.activation:
      h = self.activation(h)

    if 0.0 < self.dropout_rate <1.0:
      def dropped_inputs():
        return K.dropout(h, self.dropout_rate, K.shape(h))
      h = K.in_train_phase(dropped_inputs, h, training=training)

    y = K.dot(h, self.W2)
    if self.use_bias:
      y = K.bias_add(y, self.b2)
    return y

