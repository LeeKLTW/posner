# -*- coding: utf-8 -*-
"""
.. module:: position embedding
   :synopsis:

.. moduleauthor:: LeeKLTW

"""
from tensorflow import keras
from tensorflow.keras import backend as K

class PositionEmbedding(keras.layers.Layer):
  """Turn integers (positions) into dense vectors of fixed size.

  Expand mode: negative integers (relative position) could be used in this mode.
    - Input shape
      2D tensor with shape: `(batch_size, sequence_length)`.

    - Output shape
      3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

  Add mode:
    - Input shape
      3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    - Output shape
      3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

  Concat mode:
    - Input shape
      3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    - Output shape
      3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.

  """
  MODE_EXPAND = 'expand'
  MODE_ADD = 'add'
  MODE_CONCAT = 'concat'

  def __init__(self,
               input_dim,
               output_dim,
               mode=MODE_EXPAND,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               **kwargs):
    """Initialize position embedding layer

    Args:
      input_dim: The maximum absolute value of positions.
      output_dim: The embedding dimension.
      mode:  'expand', 'add', 'concat'
      embeddings_initializer: Initializer for linear mappings.
      embeddings_regularizer: Regularizer for linear mappings.
      embeddings_constraint: Constraints for linear mappings.
      mask_zero: The index that represents padding. Only works in `expand` mode.

    """
    super(PositionEmbedding, self).__init__(**kwargs)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.mode = mode
    self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
    self.mask_zero = mask_zero


  def get_config(self):
    config ={
      "input_dim":self.input_dim,
      "output_dim":self.output_dim,
      "mode":self.mode,
      "embeddings_initializer":self.embeddings_initializer,
      "embeddings_regularizer":self.embeddings_regularizer,
      "embeddings_constraint":self.embeddings_constraint,
      "mask_zero":self.mask_zero,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items())+list(config.items()))

  def build(self, input_shape):
    if self.mode == self.MODE_EXPAND:
      self.embeddings = self.add_weight(
        shape=(self.input_dim*2+1, self.output_dim),
        initializer=self.embeddings_initializer,
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
        name="Position_Embeddings",
      )
    else:
      self.embeddings = self.add_weight(
        shape=(self.input_dim, self.output_dim),
        initializer=self.embeddings_initializer,
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
        name="Position_Embeddings",
      )
    super(PositionEmbedding,self).build(input_shape)

  def compute_mask(self, inputs, mask=None):
    if self.mode == self.MODE_EXPAND:
      if self.mask_zero:
        output_mask = K.not_equal(inputs, self.mask_zero)
      else:
        output_mask = None
    else:
      output_mask = mask
    return output_mask

  def compute_output_shape(self,input_shape):
    if self.mode == self.MODE_EXPAND:
      return input_shape + (self.output_dim,)
    if self.mode == self.MODE_CONCAT:
      return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
    return input_shape

  #TODO: verify the formula
  def call(self, inputs, **kwargs):
    pass

