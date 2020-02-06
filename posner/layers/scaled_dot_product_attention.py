# -*- coding: utf-8 -*-
"""
.. module:: scaled_dot_product_attention
   :synopsis:

.. moduleauthor:: LeeKLTW

"""

from tensorflow import keras
from tensorflow.keras import backend as K


class ScaledDotProductAttention(keras.layers.Layer):
  r"""The attention layer that takes three inputs representing queries, keys and values.

      \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V

      See: https://arxiv.org/pdf/1706.03762.pdf
  """

  def __init__(self,
               return_attention=False,
               **kwargs, ):
    """Initialize the layer.

    Args:
      return_attention: Whether to return attention weights.
      **kwargs: Arguments for parent class.
    """
    super(ScaledDotProductAttention, self).__init__(**kwargs)
    self.support_masking = True
    self.return_attention = return_attention

  def get_config(self):
    config = {
      "return_attention": self.return_attention, }
    base_config = super(ScaledDotProductAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      query_shape, key_shape, value_shape = input_shape
    else:
      query_shape = key_shape = value_shape = input_shape
    output_shape = query_shape[:-1] + value_shape[-1:]
    if self.return_attention:
      attention_shape = query_shape[:2] + (key_shape[1],)
      return [output_shape, attention_shape]
    return output_shape

  def compute_mask(self, inputs, mask=None):
    if isinstance(mask, list):
      mask = mask[0]
    if self.return_attention:
      return [mask, None]
    return mask

  def call(self, inputs, mask=None, **kwargs):
    if isinstance(inputs, list):
      query, key, value = inputs
    else:
      query = key = value = inputs

    if isinstance(mask, list):
      mask = mask[1]
    feature_dim = K.shape(query)[-1]
    e = K.batch_dot(query, key) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
    e = K.exp(e - K.max(e, axis=-1, keepdims=True))  # prepare softmax
    if mask is not None:
      e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
    a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())  # finish softmax
    v = K.batch_dot(a, value)
    if self.return_attention:
      return [v, a]
    return v
