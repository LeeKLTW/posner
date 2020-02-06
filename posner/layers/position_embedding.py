# -*- coding: utf-8 -*-
"""
.. module:: position embedding
   :synopsis:

.. moduleauthor:: LeeKLTW

"""
from tensorflow import keras
from tensorflow.keras import backend as K


class PositionEmbedding(keras.layers.Layer):
  """Position embedding use sine and cosine functions.

    See: https://arxiv.org/pdf/1706.03762

    Expand mode:
      # Input shape
        2D tensor with shape: `(batch_size, sequence_length)`.

      # Output shape
        3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Add mode:
      # Input shape
        3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

      # Output shape
        3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
      # Input shape
        3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

      # Output shape
        3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
  MODE_EXPAND = 'expand'
  MODE_ADD = 'add'
  MODE_CONCAT = 'concat'

  def __init__(self,
               mode=MODE_ADD,
               output_dim=None,
               **kwargs):
    """
    Args:
      mode: 'expand', 'add', 'concat'
      output_dim: The embedding dimension.
      **kwargs:
    """
    if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
      if output_dim is None:
        raise NotImplementedError(
          '`output_dim` is required in `%s` mode' % mode)
      if output_dim % 2 != 0:
        raise NotImplementedError(
          'It does not make sense to use an odd output dimension: %d' % output_dim)
    super(PositionEmbedding, self).__init__(**kwargs)
    self.mode = mode
    self.output_dim = output_dim
    self.supports_masking = True

  def get_config(self):
    config = {
      'mode': self.mode,
      'output_dim': self.output_dim,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_mask(self, inputs, mask=None):
    return mask

  def compute_output_shape(self, input_shape):
    if self.mode == self.MODE_EXPAND:
      return input_shape + (self.output_dim,)
    if self.mode == self.MODE_CONCAT:
      return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
    return input_shape

  def call(self, inputs, mask=None):
    input_shape = K.shape(inputs)
    if self.mode == self.MODE_ADD:
      batch_size, seg_len, output_dim = input_shape[0],input_shape[1],input_shape[2]
      pos_input = K.tile(K.expand_dims(K.arange(0, seg_len), axis=0), [batch_size,1])
    elif self.mode == self.MODE_CONCAT:
      batch_size, seg_len, output_dim = input_shape[0],input_shape[1],self.output_dim
      pos_input = K.tile(K.expand_dims(K.arange(0, seg_len), axis=0), [batch_size,1])
    else:
      output_dim = self.output_dim
      pos_input = inputs
    if K.dtype(pos_input) != K.floatx():
      pos_input = K.cast(pos_input, K.floatx())
    evens = K.arange(0, output_dim//2) * 2
    odds = K.arange(0, output_dim//2) * 2 + 1

    even_embedding = K.sin(
      K.dot(
        K.expand_dims(pos_input,-1),
        K.expand_dims(
          1.0/K.pow(10000.0,K.cast(evens,K.floatx())/
                                K.cast(output_dim,K.floatx())),0))
    )

    odd_embedding = K.cos(
      K.dot(
        K.expand_dims(pos_input,-1),
        K.expand_dims(
          1.0/K.pow(10000.0,K.cast(odds-1,K.floatx())/
                                K.cast(output_dim,K.floatx())),0))
    )
    embedding = K.stack([even_embedding, odd_embedding], axis=-1)
    output = K.reshape(embedding, [-1, K.shape(inputs)[1], output_dim])
    if self.mode == self.MODE_CONCAT:
      output = K.concatenate([inputs, output], axis=-1)
    if self.mode == self.MODE_ADD:
      output += inputs
    return output
