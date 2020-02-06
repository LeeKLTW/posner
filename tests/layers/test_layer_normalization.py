# -*- coding: utf-8 -*-
import unittest
import numpy as np
from tensorflow import keras
from posner.layers import LayerNormalization


class TestLayerNormalization(unittest.TestCase):

  def test_sample(self):
    input_layer = keras.layers.Input(
      shape=(2, 3),
      name='Input',)
    norm_layer = LayerNormalization(
      name='Layer-Normalization',
    )(input_layer)
    model = keras.models.Model(
      inputs=input_layer,
      outputs=norm_layer,)
    model.compile(
      optimizer='adam',
      loss='mse',
      metrics={},)
    model.summary()

    inputs = np.array([[
      [0.2, 0.1, 0.3],
      [0.5, 0.1, 0.1],
    ]])
    predict = model.predict(inputs)
    expected = np.asarray([[
      [0.0, -1.22474487, 1.22474487],
      [1.41421356, -0.707106781, -0.707106781],
    ]])
    self.assertTrue(np.allclose(expected, predict))

    input_layer = keras.layers.Input(
      shape=(10, 256),
      name='Input',)
    norm_layer = LayerNormalization(
      name='Layer-Normalization',
      beta_initializer='ones',
    )(input_layer)
    model = keras.models.Model(
      inputs=input_layer,
      outputs=norm_layer,)
    model.compile(
      optimizer='adam',
      loss='mse',
      metrics={},)
    model.summary()

    inputs = np.zeros((2, 10, 256))
    predict = model.predict(inputs)
    expected = np.ones((2, 10, 256))
    self.assertTrue(np.allclose(expected, predict))

  # TODO: add more test cases