# -*- coding: utf-8 -*-
import unittest
from tensorflow import keras
from posner.layers.bert_layers import get_inputs, get_embedding, \
  TokenEmbedding, TaskEmbedding, EmbeddingSimilarity, Masked,Extract

class TestGetInput(unittest.TestCase):
  def test_name(self):
    inputs = get_inputs(seq_len=512)
    self.assertEqual(3, len(inputs))
    self.assertTrue("Segment" in inputs[1].name )

class TestGetEmbedding(unittest.TestCase):
  def test_sample(self):
    inputs = get_inputs(seq_len=512)
    embed_layer = get_embedding(inputs, token_num=12, pos_num=512, embed_dim=768)
    model = keras.models.Model(inputs=inputs,outputs=embed_layer)
    model.compile(optimizer='adam', loss='mse', metrics={})
    model.summary()
    self.assertEqual((None, 512, 768), model.layers[-1].output_shape)

  def test_no_dropout(self):
    inputs = get_inputs(seq_len=512)
    embed_layer = get_embedding(inputs, token_num=12, pos_num=512, embed_dim=768, dropout_rate=0.0)
    model = keras.models.Model(inputs=inputs, outputs=embed_layer)
    model.compile(optimizer='adam', loss='mse', metrics={})
    model.summary()
    self.assertEqual((None, 512, 768), model.layers[-1].output_shape)
