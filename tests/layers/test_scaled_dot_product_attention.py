# -*- coding: utf-8 -*-
import unittest
import numpy as np
from tensorflow import keras
from posner.layers import ScaledDotProductAttention


class TestAttention(unittest.TestCase):

  def test_sample(self):
    input_layer = keras.layers.Input(
      shape=(5,),
      name='Input', )
    embed_layer = keras.layers.Embedding(
      input_dim=4,  # input token [0,1,2,3] for embedding
      output_dim=5,
      mask_zero=True,
      weights=[
        np.array([
          [0.1, 0.2, 0.3, 0.4, 0.5],  # token 0
          [0.2, 0.3, 0.4, 0.6, 0.5],  # token 1
          [0.4, 0.7, 0.2, 0.6, 0.9],  # token 2
          [0.3, 0.5, 0.8, 0.9, 0.1],  # token 3
        ]),
      ],
      name='Embedding',
    )(input_layer)
    att_layer = ScaledDotProductAttention(name='Attention')(embed_layer)
    model = keras.models.Model(inputs=input_layer, outputs=att_layer)
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    inputs = np.array([[1, 2, 3, 1, 0]])
    predict = model.predict(inputs)[0]
    print("Sample prediction: \n", predict)
    self.assertTrue(np.allclose(predict[0], predict[3]))  # token 1 & token 1
    self.assertTrue(np.allclose(
      np.asarray([0.27250203, 0.44500405, 0.45269585, 0.6751187, 0.49476662]),
      predict[2]))
