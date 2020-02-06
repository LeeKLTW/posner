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
    :param output_dim: The embedding dimension.
    :param kwargs:
    """
    if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
      if output_dim is None:
        raise NotImplementedError(
          '`output_dim` is required in `%s` mode' % mode)
      if output_dim % 2 != 0:
        raise NotImplementedError(
          'It does not make sense to use an odd output dimension: %d' % output_dim)
    self.mode = mode
    self.output_dim = output_dim
    self.supports_masking = True
    super(PositionEmbedding, self).__init__(**kwargs)

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
