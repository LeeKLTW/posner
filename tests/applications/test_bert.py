# -*- coding: utf-8 -*-
import unittest
import os
import tempfile
import numpy as np
from tensorflow import keras

from posner.applications.bert import get_custom_objects, get_model, \
  load_trained_model_from_checkpoint


class TestBERT(unittest.TestCase):

  def test_sample(self):
    model = get_model(
      token_num=200,
      head_num=3,
      transformer_num=2,
    )
    # # FIXME: TypeError: __init__() missing 1 required positional argument: 'index'
    # model_path = os.path.join(tempfile.gettempdir(),
    #                           'keras_bert_%f.h5' % np.random.random())
    # model.save(model_path)
    # model = keras.models.load_model(
    #   model_path,
    #   custom_objects=get_custom_objects(),
    # )
    model.summary(line_length=200)

  def test_task_embed(self):
    inputs, outputs = get_model(
      token_num=20,
      embed_dim=12,
      head_num=3,
      transformer_num=2,
      use_task_embed=True,
      task_num=10,
      training=False,
      dropout_rate=0.0,
    )
    model = keras.models.Model(inputs, outputs)
    model_path = os.path.join(tempfile.gettempdir(),
                              'keras_bert_%f.h5' % np.random.random())
    model.save(model_path)
    model = keras.models.load_model(
      model_path,
      custom_objects=get_custom_objects(),
    )
    model.summary(line_length=200)


class TestLoader(unittest.TestCase):

  def test_load_trained(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False)
    model.summary()

  def test_load_trained_shorter(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False, seq_len=8)
    model.summary()

  def test_load_training(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=True)
    model.summary()

  def test_load_adapter(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(
      config_path,
      model_path,
      training=False,
      use_adapter=True,
      trainable=['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i
                 in range(2)] +
                ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in
                 range(2)] +
                ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in
                 range(2)] +
                ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(2)],
    )
    model.summary()

  def test_load_output_layer_num(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False,
                                               output_layer_num=4)
    model.summary()
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False,
                                               output_layer_num=[0])
    model.summary()
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False,
                                               output_layer_num=[1])
    model.summary()
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False,
                                               output_layer_num=[-1])
    model.summary()
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False,
                                               output_layer_num=[-2])
    model.summary()
    model = load_trained_model_from_checkpoint(config_path, model_path,
                                               training=False,
                                               output_layer_num=[0, -1])
    model.summary()

  def test_load_with_trainable_prefixes(self):
    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(
      config_path,
      model_path,
      training=False,
      trainable=['Encoder'],
    )
    model.summary()

  def test_on_chinese_daily_ner(self):
    import numpy as np
    from posner.datasets import chinese_daily_ner
    from posner.utils.bert_tokenization import FullTokenizer

    (x_train, y_train), (x_test, y_test), (vocab, pos_tags) = \
      chinese_daily_ner.load_data(path=None, maxlen=16, onehot=True, min_freq=2)

    current_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_path, 'test_checkpoint',
                               'bert_config.json')
    model_path = os.path.join(current_path, 'test_checkpoint',
                              'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(
      config_path,
      model_path,
      training=False,
      trainable=['Encoder'],
    )
    model.summary(line_length=120)
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    tokenizer =FullTokenizer(os.path.join(current_path, 'test_checkpoint',
                               'vocab.txt'))

    # text = 'all language'
    # x = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    x = np.zeros((1,16))
    x[0,0]=1
    y = np.zeros((1, 16, 4))
    y[0, 0, 0] = 1

    model.fit([x, np.zeros((1, 16)).reshape(1, 16)], y,epochs=100)


