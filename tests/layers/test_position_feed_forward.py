# -*- coding: utf-8 -*-
import unittest
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from posner.layers import PositionFeedForward

class TestPositionFeedForward(unittest.TestCase):

  @staticmethod
  def _leaky_relu(x):
    return keras.activations.relu(x, alpha=0.01)

  def test_sample(self):
    input_layer = keras.layers.Input(
      shape=(1, 3),
      name='Input',
    )
    feed_forward_layer = PositionFeedForward(
      units=4,
      activation=self._leaky_relu,
      weights=[
        np.asarray([
          [0.1, 0.2, 0.3, 0.4],
          [-0.1, 0.2, -0.3, 0.4],
          [0.1, -0.2, 0.3, -0.4],
        ]),
        np.asarray([
          0.0, -0.1, 0.2, -0.3,
        ]),
        np.asarray([
          [0.1, 0.2, 0.3],
          [-0.1, 0.2, -0.3],
          [0.1, -0.2, 0.3],
          [-0.1, 0.2, 0.3],
        ]),
        np.asarray([
          0.0, 0.1, -0.2,
        ]),
      ],
      name='PositionFeedForward',
    )(input_layer)
    model = keras.models.Model(
      inputs=input_layer,
      outputs=feed_forward_layer,
    )
    model.compile(
      optimizer='adam',
      loss='mse',
      metrics={},
    )
    model.summary()
    inputs = np.array([[[0.2, 0.1, 0.3]]]) #shape (1,1,3)
    predict = model.predict(inputs)
    expected = np.asarray([[[0.0364, 0.0432, -0.0926]]]) #shape (1,1,3)
    self.assertTrue(np.allclose(expected, predict), predict)
