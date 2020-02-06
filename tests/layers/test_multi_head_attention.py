# -*- coding: utf-8 -*-
import unittest

import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K

from posner.layers import MultiHeadAttention


class GetMask(keras.layers.Layer):
  """Capture mask value for testing."""

  def __init__(self, **kwargs):
    super(GetMask, self).__init__(**kwargs)
    self.support_masking = True

  def compute_output_shape(self, input_shape):
    return input_shape[:-1]

  def call(self, inputs, mask=None, **kwargs):
    return K.cast(mask, K.floatx())


class TestMultiHead(unittest.TestCase):

  def test_sample(self):
    input_layer = keras.layers.Input(
      shape=(512,),
      name='Input',
    )
    embed_layer = keras.layers.Embedding(
      input_dim=12,
      output_dim=768,
      mask_zero=True,
      name='Embedding',
    )(input_layer)
    output_layer = MultiHeadAttention(
      head_num=12,
      name='Multi-Head',
    )(embed_layer)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(
      optimizer='adam',
      loss='mse',
      metrics={},
    )
    model.summary()

  def test_invalid_head_num(self):
    with self.assertRaises(IndexError):
      input_layer = keras.layers.Input(
        shape=(2, 3),
        name='Input',
      )
      MultiHeadAttention(
        head_num=2,
        name='Multi-Head',
      )(input_layer)

  def test_mask_single(self):
    input_layer = keras.layers.Input(shape=(None,))
    embed_layer = keras.layers.Embedding(input_dim=3, output_dim=4,
                                         mask_zero=True)(input_layer)
    att_layer = MultiHeadAttention(
      head_num=2,
      name='Multi-Head-2',
    )(embed_layer)
    mask_layer = GetMask()(att_layer)
    model = keras.models.Model(inputs=input_layer, outputs=mask_layer)
    model.compile(optimizer='adam', loss='mse', metrics={})
    predicts = model.predict(np.asarray([[1, 2, 1, 2, 0, 0]])).tolist()
    self.assertEqual([1.0] * 4 + [0.0] * 2, predicts[0], predicts[0])

  def test_mask_multi(self):
    input_q_layer = keras.layers.Input(shape=(None,))
    input_kv_layer = keras.layers.Input(shape=(None,))
    embed_q_layer = keras.layers.Embedding(input_dim=3, output_dim=4,
                                           mask_zero=True)(input_q_layer)
    embed_kv_layer = keras.layers.Embedding(input_dim=3, output_dim=4,
                                            mask_zero=True)(input_kv_layer)
    att_layer = MultiHeadAttention(
      head_num=2,
      name='Multi-Head-2',
    )([embed_q_layer, embed_kv_layer, embed_kv_layer])
    mask_layer = GetMask()(att_layer)
    model = keras.models.Model(inputs=[input_q_layer, input_kv_layer],
                               outputs=mask_layer)
    model.compile(optimizer='adam', loss='mse', metrics={})
    predicts = model.predict([np.asarray([[1, 2, 1, 2, 0, 0]]),
                              np.asarray([[1, 2, 2, 0, 0, 0]])]).tolist()
    self.assertEqual([1.0] * 4 + [0.0] * 2, predicts[0], predicts[0])
