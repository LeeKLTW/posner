# -*- coding: utf-8 -*-

from tensorflow import keras
from posner.layers import CRF
from posner.applications import bert

layer_num = 12

def load_trained_model_from_checkpoint(
        config_file,
        checkpoint_file,
        crf_dims,
        training=False,
        trainable=None,
        output_layer_num=1,
        seq_len=int(1e9),
        **kwargs):

  model = bert.load_trained_model_from_checkpoint(
    config_file,
    checkpoint_file,
    training=training,  # MLM, NSP
    use_adapter=True,
    trainable=  # Adapter
    ['Encoder-{}-MultiHeadSelfAttention-Adapter'.format(i + 1) for i in
     range(layer_num)] +
    ['Encoder-{}-FeedForward-Adapter'.format(i + 1) for i in range(layer_num)] +
    ['Encoder-{}-MultiHeadSelfAttention-Norm'.format(i + 1) for i in
     range(layer_num)] +
    ['Encoder-{}-FeedForward-Norm'.format(i + 1) for i in range(layer_num)],
  )
  crf = CRF(crf_dims, name='CRF')
  inp = model.input
  out = crf(model.layers[-9].output)
  model = keras.models.Model(inp, out)
  model.summary(line_length=150)
  return model
