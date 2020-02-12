# -*- coding: utf-8 -*-
"""
.. module:: bert
   :synopsis:

.. moduleauthor:: LeeKLTW

"""
import json
import six
import copy

import numpy as np
import tensorflow as tf
from tensorflow import keras
from ..activations import gelu
from ..layers import LayerNormalization
from ..optimizers import AdamWarmup
from ..layers.bert_layers import get_inputs, get_embedding, \
  TokenEmbedding, TaskEmbedding, EmbeddingSimilarity, Masked, Extract, \
  PositionEmbedding

from .transformer import get_encoders
from .transformer import get_custom_objects as get_encoder_custom_objects

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def get_model(token_num,
              pos_num=512,
              seq_len=512,
              embed_dim=768,
              transformer_num=12,
              head_num=12,
              feed_forward_dim=3072,
              dropout_rate=0.1,
              attention_activation=None,
              feed_forward_activation='gelu',
              training=True,
              trainable=None,
              output_layer_num=1,
              use_task_embed=False,
              task_num=10,
              use_adapter=False,
              adapter_units=None):
  """Get BERT model.
  See: https://arxiv.org/pdf/1810.04805.pdf
  Also: https://github.com/tensorﬂow/tensor2tensor

  Args:
    token_num: Number of tokens.
    pos_num: Maximum position.
    seq_len: Maximum length of the input sequence or None.
    embed_dim: Dimensions of embeddings.
    transformer_num: Number of transformers.
    head_num: Number of heads in multi-head attention in each transformer.
    feed_forward_dim: Dimension of the feed forward layer in each transformer.
    dropout_rate: Dropout rate.
    attention_activation: Activation for attention layers.
    feed_forward_activation: Activation for feed-forward layers.
    training: A built model with MLM and NSP outputs will be returned if it is
    `True`,otherwise the input layers and the last feature extraction layer will
    be returned.
    trainable: Whether the model is trainable.
    output_layer_num: The number of layers whose outputs will be concatenated as
     a single output. Only available when `training` is `False`.
    use_task_embed: Whether to add task embeddings to existed embeddings.
    task_num:The number of tasks.
    use_adapter: Whether to use feed-forward adapters before each residual
    connections.
    adapter_units:The dimension of the first transformation in feed-forward
    adapter.

  Returns:
    The built model.
  """
  if attention_activation == "gelu":
    attention_activation = gelu

  if feed_forward_activation == "gelu":
    feed_forward_activation = gelu

  if trainable is None:
    trainable = training

  if adapter_units is None:
    adapter_units = max(1, embed_dim // 100)

  def _trainable(_layer):
    if isinstance(trainable, (list, tuple, set)):
      for prefix in trainable:
        if _layer.name.startswith(prefix):
          return True
        return False
    return trainable

  inputs = get_inputs(seq_len=seq_len)
  embed_layer, embed_weights = get_embedding(inputs,
                                             token_num=token_num,
                                             embed_dim=embed_dim,
                                             pos_num=pos_num,
                                             dropout_rate=dropout_rate,
                                             )
  if use_task_embed:
    task_input = keras.layers.Input(shape=(1,),
                                    name="Input-Task",
                                    )
    embed_layer = TaskEmbedding(
      input_dim=task_num,
      output_dim=embed_dim,
      mask_zero=False,
      name="Embedding-Task",
    )([embed_layer, task_input])
    inputs = inputs[:2] + [task_input, inputs[-1]]

  if dropout_rate > 0.0:
    dropout_layer = keras.layers.Dropout(
      rate=dropout_rate,
      name="Embedding-Dropout",
    )(embed_layer)
  else:
    dropout_layer = embed_layer

  embed_layer = LayerNormalization(
    trainable=trainable,
    name="Embedding-Norm",
  )(dropout_layer)
  transformed = get_encoders(
    encoder_num=transformer_num,
    input_layer=embed_layer,
    head_num=head_num,
    hidden_dim=feed_forward_dim,
    attention_activation=attention_activation,
    feed_forward_activation=feed_forward_activation,
    dropout_rate=dropout_rate,
    use_adapter=use_adapter,
    adapter_units=adapter_units,
    adapter_activation=gelu,
  )

  if training:
    mlm_dense_layer = keras.layers.Dense(
      units=embed_dim,
      activation=feed_forward_activation,
      name="MLM-Dense",
    )(transformed)
    mlm_norm_layer = LayerNormalization(name="MLM-Norm")(mlm_dense_layer)
    mlm_pred_layer = EmbeddingSimilarity(name="MLM-Sim")(
      [mlm_norm_layer, embed_weights]
    )
    masked_layer = Masked(name="MLM")([mlm_pred_layer, inputs[-1]])
    extract_layer = Extract(index=0, name="Extract")(transformed)
    nsp_dense_layer = keras.layers.Dense(
      units=embed_dim,
      activation='tanh',
      name="NSP-Dense",
    )(extract_layer)
    nsp_predict_layer = keras.layers.Dense(
      units=2,
      activation="softmax",
      name="NSP",
    )(nsp_dense_layer)
    model = keras.models.Model(inputs=inputs,
                               outputs=[masked_layer, nsp_predict_layer])
    for layer in model.layers:
      layer.trainable = _trainable(layer)
    return model
  else:
    if use_task_embed:
      inputs = inputs[:3]
    else:
      inputs = inputs[:2]
    model = keras.models.Model(inputs=inputs, outputs=transformed)
    for layer in model.layers:
      layer.trainable = _trainable(layer)
    if isinstance(output_layer_num, int):
      output_layer_num = min(output_layer_num, transformer_num)
      output_layer_num = [-i for i in range(1, output_layer_num + 1)]
    outputs = []
    for layer_index in output_layer_num:
      if layer_index < 0:
        layer_index = transformer_num + layer_index
      layer_index += 1
      layer = model.get_layer(name=f"Encoder-{layer_index}-FeedForward-Norm")
      outputs.append(layer.output)
    if len(outputs) > 1:
      transformed = keras.layers.Concatenate(
        name="Encoder-Output"
      )(list(reversed(outputs)))
    else:
      transformed = outputs[0]
    return inputs, transformed


def build_model_from_config(config_file,
                            training=False,
                            trainable=None,
                            output_layer_num=1,
                            seq_len=int(1e9),
                            **kwargs):
  """Build the model from config file.

  Args:
    config_file: The path to the JSON configuration file.
    training: If training, the whole model will be returned. Otherwise, the MLM and NSP parts will be ignored.
    trainable: Whether the model is trainable.
    output_layer_num: The number of layers whose outputs will be concatenated as a single output. Only available when `training` is `False`.
    seq_len: If it is not None and it is shorter than the value in the config file, the weights in position embeddings will be sliced to fit the new length.
    **kwargs:

  Returns:
    keras.models.Model

  """
  with open(config_file, 'r') as reader:
    config = json.loads(reader.read())
  if seq_len is not None:
    config['max_position_embeddings'] = seq_len = min(seq_len, config[
      'max_position_embeddings'])
  if trainable is None:
    trainable = training
  model = get_model(
    token_num=config['vocab_size'],
    pos_num=config['max_position_embeddings'],
    seq_len=seq_len,
    embed_dim=config['hidden_size'],
    transformer_num=config['num_hidden_layers'],
    head_num=config['num_attention_heads'],
    feed_forward_dim=config['intermediate_size'],
    feed_forward_activation=config['hidden_act'],
    training=training,
    trainable=trainable,
    output_layer_num=output_layer_num,
    **kwargs)
  if not training:
    inputs, outputs = model
    model = keras.models.Model(inputs=inputs, outputs=outputs)
  return model, config


def load_model_weights_from_checkpoint(model,
                                       config,
                                       checkpoint_file,
                                       training=False):
  """Load trained official model from checkpoint.

  Args:
    model: Built keras model.
    config: Loaded configuration file.
    checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    training: If training, the whole model will be returned. Otherwise, the MLM and NSP parts will be ignored.

  Returns:

  """

  def checkpoint_loader(checkpoint_file):
    def _loader(name):
      return tf.train.load_variable(checkpoint_file, name)

    return _loader

  loader = checkpoint_loader(checkpoint_file)

  model.get_layer(name='Embedding-Token').set_weights([
    loader('bert/embeddings/word_embeddings'),
  ])
  model.get_layer(name='Embedding-Position').set_weights([
    loader('bert/embeddings/position_embeddings')[
    :config['max_position_embeddings'], :],
  ])
  model.get_layer(name='Embedding-Segment').set_weights([
    loader('bert/embeddings/token_type_embeddings'),
  ])
  model.get_layer(name='Embedding-Norm').set_weights([
    loader('bert/embeddings/LayerNorm/gamma'),
    loader('bert/embeddings/LayerNorm/beta'),
  ])
  for i in range(config['num_hidden_layers']):
    try:
      model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
    except ValueError as e:
      continue
    model.get_layer(
      name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
      loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
      loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
      loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
      loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
      loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
      loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
      loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
      loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
    ])
    model.get_layer(
      name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
      loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
      loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
    ])
    model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
      loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
      loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
      loader('bert/encoder/layer_%d/output/dense/kernel' % i),
      loader('bert/encoder/layer_%d/output/dense/bias' % i),
    ])
    model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
      loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
      loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
    ])
  if training:
    model.get_layer(name='MLM-Dense').set_weights([
      loader('cls/predictions/transform/dense/kernel'),
      loader('cls/predictions/transform/dense/bias'),
    ])
    model.get_layer(name='MLM-Norm').set_weights([
      loader('cls/predictions/transform/LayerNorm/gamma'),
      loader('cls/predictions/transform/LayerNorm/beta'),
    ])
    model.get_layer(name='MLM-Sim').set_weights([
      loader('cls/predictions/output_bias'),
    ])
    model.get_layer(name='NSP-Dense').set_weights([
      loader('bert/pooler/dense/kernel'),
      loader('bert/pooler/dense/bias'),
    ])
    model.get_layer(name='NSP').set_weights([
      np.transpose(loader('cls/seq_relationship/output_weights')),
      loader('cls/seq_relationship/output_bias'),
    ])


def load_trained_model_from_checkpoint(config_file,
                                       checkpoint_file,
                                       training=False,
                                       trainable=None,
                                       output_layer_num=1,
                                       seq_len=int(1e9),
                                       **kwargs):
  """Load trained official model from checkpoint.

  Args:
    config_file: The path to the JSON configuration file.
    checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    training: If training, the whole model will be returned. Otherwise, the MLM and NSP parts will be ignored.
    trainable: Whether the model is trainable. The default value is the same with `training`.
    output_layer_num: The number of layers whose outputs will be concatenated as a single output.Only available when `training` is `False`.
    seq_len: If it is not None and it is shorter than the value in the config file, the weights in position embeddings will be sliced to fit the new length.
    **kwargs:

  Returns:
    keras.model
  """
  model, config = build_model_from_config(
    config_file,
    training=training,
    trainable=trainable,
    output_layer_num=output_layer_num,
    seq_len=seq_len,
    **kwargs, )
  load_model_weights_from_checkpoint(
    model,
    config,
    checkpoint_file,
    training=training)
  return model


def get_custom_objects():
  """Get all custom objects for loading saved models."""
  custom_objects = get_encoder_custom_objects()
  custom_objects['PositionEmbedding'] = PositionEmbedding
  custom_objects['TokenEmbedding'] = TokenEmbedding
  custom_objects['EmbeddingSimilarity'] = EmbeddingSimilarity
  custom_objects['TaskEmbedding'] = TaskEmbedding
  custom_objects['Masked'] = Masked
  custom_objects['Extract'] = Extract
  custom_objects['gelu'] = gelu
  custom_objects['AdamWarmup'] = AdamWarmup
  return custom_objects
