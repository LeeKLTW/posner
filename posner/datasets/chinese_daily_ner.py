# -*- coding: utf-8 -*-
import os
from collections import Counter

import numpy as np
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences

from absl import logging

logging.set_verbosity(logging.INFO)


def _parse_data(file_path,
                text_index=0,
                label_index=1):
  x_data, y_data = [], []
  with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()
    x, y = [], []
    for line in lines:
      rows = line.split(' ')
      if len(rows) == 1:
        x_data.append(x)
        y_data.append(y)
        x = []
        y = []
      else:
        x.append(rows[text_index])
        y.append(rows[label_index])
  return x_data, y_data


def _make_dir(path):
  if not os.path.isdir(path):
    try:
      os.remove(path)
    except:
      pass
    os.mkdir(path)


def _process_data(data, character, pos_tags, onehot, maxlen=None):
  if maxlen is None:
    maxlen = max(len(s) for s in data[0])

  word2idx = dict((w, i) for i, w in enumerate(character))

  x = []
  for s in data[0]:
    temp = [word2idx.get(c, 1) for c in s]
    x.append(temp)

  y = []
  for s in data[1]:
    temp = [pos_tags.index(p) for p in s]
    y.append(temp)

  x = pad_sequences(x, maxlen)
  y = pad_sequences(y, maxlen, value=-1)

  if onehot:
    y = np.eye(len(pos_tags), dtype='float32')[y]
  else:
    y = np.expand_dims(y, 2)
  return x, y


def load_data(path=None, maxlen=None, onehot=True, min_freq=2):
  """Loads the Chinese Daily NER dataset for NER Labelling.

  Arguments
      path: where to cache the data (relative to `~/.keras/dataset`).
      maxlen: truncate sequences after this length.
      onehot: (bool) NER Label to be one-hot encoding
      min_freq: minimum frequency for character.

  # Returns
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test),(character, pos_tags)`.

  """


  if path is None:
    path = os.path.join(os.path.expanduser('~'), '.keras', 'chinese_daily_ner')
  _make_dir(path)
  train_path = os.path.join(path, 'example.train')
  test_path = os.path.join(path, 'example.test')

  train_path = get_file(
    train_path,
    origin='https://raw.githubusercontent.com/zjy-ucas/ChineseNER/master/data/example.train',
  )
  test_path = get_file(
    test_path,
    origin='https://raw.githubusercontent.com/zjy-ucas/ChineseNER/master/data/example.test')

  logging.info("Download Chinese Daily NER save at folder:\n {} ".format(path))

  train = _parse_data(train_path)
  test = _parse_data(test_path)

  word_counts = Counter(
    row[0].lower() for sample in train[0] + test[0] for row in sample)

  character = ['<pad>', '<unk>']
  character += [w for w, f in iter(word_counts.items()) if f >= min_freq]

  flattened_list = [y for x in train[1] + test[1] for y in x]
  pos_tags = sorted(set(flattened_list))

  x_train, y_train = _process_data(train, character, pos_tags, onehot,
                                   maxlen=maxlen)
  x_test, y_test = _process_data(test, character, pos_tags, onehot, maxlen=maxlen)
  return (x_train, y_train), (x_test, y_test), (character, pos_tags)
