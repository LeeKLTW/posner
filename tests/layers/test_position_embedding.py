# -*- coding: utf-8 -*-
import os
import tempfile
import unittest

import numpy as np

from tensorflow import keras
from posner.layers import PositionEmbedding


class TestSinCosPosEmbd(unittest.TestCase):

  def test_invalid_output_dim(self):
    with self.assertRaises(NotImplementedError):
      PositionEmbedding(
        mode=PositionEmbedding.MODE_EXPAND,
        output_dim=5,
      )

  def test_missing_output_dim(self):
    with self.assertRaises(NotImplementedError):
      PositionEmbedding(
        mode=PositionEmbedding.MODE_EXPAND,
      )

  def test_add(self):
    seq_len = np.random.randint(1, 10)
    embed_dim = np.random.randint(1, 20) * 2
    inputs = np.ones((1, seq_len, embed_dim))
    model = keras.models.Sequential()
    model.add(PositionEmbedding(
      input_shape=(seq_len, embed_dim),
      mode=PositionEmbedding.MODE_ADD,
      name='Pos-Embd',
    ))
    model.compile('adam', 'mse')
    model_path = os.path.join(tempfile.gettempdir(),
                              'pos_embd_%f.h5' % np.random.random())
    model.save(model_path)
    model = keras.models.load_model(model_path, custom_objects={
      'PositionEmbedding': PositionEmbedding})
    model.summary()
    predicts = model.predict(inputs)[0].tolist()
    for i in range(seq_len):
      for j in range(embed_dim):
        actual = predicts[i][j]
        if j % 2 == 0:
          expect = 1.0 + np.sin(i / 10000.0 ** (float(j) / embed_dim))
        else:
          expect = 1.0 + np.cos(i / 10000.0 ** ((j - 1.0) / embed_dim))
        self.assertAlmostEqual(expect, actual, places=6,
                               msg=(embed_dim, i, j, expect, actual))

  def test_concat(self):
    seq_len = np.random.randint(1, 10)
    feature_dim = np.random.randint(1, 20)
    embed_dim = np.random.randint(1, 20) * 2
    inputs = np.ones((1, seq_len, feature_dim))
    model = keras.models.Sequential()

    model.add(PositionEmbedding(
      input_shape=(seq_len, feature_dim),
      output_dim=embed_dim,
      mode=PositionEmbedding.MODE_CONCAT,
      name='Pos-Embd',
    ))
    model.compile('adam', 'mse')
    model_path = os.path.join(tempfile.gettempdir(),
                              'test_pos_embd_%f.h5' % np.random.random())
    model.save(model_path)
    model = keras.models.load_model(model_path, custom_objects={
      'PositionEmbedding': PositionEmbedding})
    model.summary()
    predicts = model.predict(inputs)[0].tolist()
    for i in range(seq_len):
      for j in range(embed_dim):
        actual = predicts[i][feature_dim + j]
        if j % 2 == 0:
          expect = np.sin(i / 10000.0 ** (float(j) / embed_dim))
        else:
          expect = np.cos(i / 10000.0 ** ((j - 1.0) / embed_dim))
        self.assertAlmostEqual(expect, actual, places=6,
                               msg=(embed_dim, i, j, expect, actual))
