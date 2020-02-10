# -*- coding: utf-8 -*-
from tensorflow import keras
from tensorflow.keras import backend as K


def get_inputs(seq_len):
  """Get input layers.
  See: https://arxiv.org/pdf/1810.04805.pdf

  Args:
    seq_len: Length of the sequence or None.

  Returns:
    list of Input
  """
  names = ["Token", "Segment", "Masked"]
  return [keras.layers.Input(
    shape=(seq_len,),
    name=f"Input-{name}",
  ) for name in names]


def get_embedding(inputs,
                  token_num,
                  pos_num,
                  embed_dim,
                  dropout_rate=0.1,
                  trainable=True):
  embeddings = [
    TokenEmbedding(
      input_dim=token_num,
      output_dim=embed_dim,
      mask_zero=True,
      trainable=trainable,
      name="Embedding-Token",
    )(inputs[0]),

    keras.layers.Embedding(
      input_dim=2,
      output_dim=embed_dim,
      trainable=trainable,
      name="Embedding-Segment",
    )(inputs[1]),
  ]
  embeddings[0], embed_weights = embeddings[0]
  embed_layer = keras.layers.Add(name="Embedding-Token-Segment")(embeddings)
  embed_layer = PositionEmbedding(
    input_dim=pos_num,
    output_dim=embed_dim,
    mode=PositionEmbedding.MODE_ADD,
    trainable=trainable,
    name='Embedding-Position',
  )(embed_layer)
  return embed_layer, embed_weights


class PositionEmbedding(keras.layers.Layer):
  """Turn integers (positions) into dense vectors of fixed size.
  eg. [[-4], [10]] -> [[0.25, 0.1], [0.6, -0.2]]

  Expand mode: negative integers (relative position) could be used in this mode.
      # Input shape
          2D tensor with shape: `(batch_size, sequence_length)`.

      # Output shape
          3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

  Add mode:
      # Input shape
          3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

      # Output shape
          3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

  Concat mode:
      # Input shape
          3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

      # Output shape
          3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
  """
  MODE_EXPAND = 'expand'
  MODE_ADD = 'add'
  MODE_CONCAT = 'concat'

  def __init__(self,
               input_dim,
               output_dim,
               mode=MODE_EXPAND,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               **kwargs):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.mode = mode
    self.embeddings_initializer = keras.initializers.get(
      embeddings_initializer)
    self.embeddings_regularizer = keras.regularizers.get(
      embeddings_regularizer)
    self.activity_regularizer = keras.regularizers.get(activity_regularizer)
    self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
    self.mask_zero = mask_zero
    self.supports_masking = mask_zero is not False

    self.embeddings = None
    super(PositionEmbedding, self).__init__(**kwargs)

  def get_config(self):
    config = {'input_dim': self.input_dim,
              'output_dim': self.output_dim,
              'mode': self.mode,
              'embeddings_initializer': keras.initializers.serialize(
                self.embeddings_initializer),
              'embeddings_regularizer': keras.regularizers.serialize(
                self.embeddings_regularizer),
              'activity_regularizer': keras.regularizers.serialize(
                self.activity_regularizer),
              'embeddings_constraint': keras.constraints.serialize(
                self.embeddings_constraint),
              'mask_zero': self.mask_zero}
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    if self.mode == self.MODE_EXPAND:
      self.embeddings = self.add_weight(
        shape=(self.input_dim * 2 + 1, self.output_dim),
        initializer=self.embeddings_initializer,
        name='embeddings',
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
      )
    else:
      self.embeddings = self.add_weight(
        shape=(self.input_dim, self.output_dim),
        initializer=self.embeddings_initializer,
        name='embeddings',
        regularizer=self.embeddings_regularizer,
        constraint=self.embeddings_constraint,
      )
    super(PositionEmbedding, self).build(input_shape)

  def compute_mask(self, inputs, mask=None):
    if self.mode == self.MODE_EXPAND:
      if self.mask_zero:
        output_mask = K.not_equal(inputs, self.mask_zero)
      else:
        output_mask = None
    else:
      output_mask = mask
    return output_mask

  def compute_output_shape(self, input_shape):
    if self.mode == self.MODE_EXPAND:
      return input_shape + (self.output_dim,)
    if self.mode == self.MODE_CONCAT:
      return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
    return input_shape

  def call(self, inputs, **kwargs):
    if self.mode == self.MODE_EXPAND:
      if K.dtype(inputs) != 'int32':
        inputs = K.cast(inputs, 'int32')
      return K.gather(
        self.embeddings,
        K.minimum(K.maximum(inputs, -self.input_dim),
                  self.input_dim) + self.input_dim,
      )
    input_shape = K.shape(inputs)
    if self.mode == self.MODE_ADD:
      batch_size, seq_len, output_dim = input_shape[0], input_shape[1], \
                                        input_shape[2]
    else:
      batch_size, seq_len, output_dim = input_shape[0], input_shape[
        1], self.output_dim
    pos_embeddings = K.tile(
      K.expand_dims(self.embeddings[:seq_len, :self.output_dim], axis=0),
      [batch_size, 1, 1],
    )
    if self.mode == self.MODE_ADD:
      return inputs + pos_embeddings
    return K.concatenate([inputs, pos_embeddings], axis=-1)


class TokenEmbedding(keras.layers.Embedding):
  def compute_output_shape(self, input_shape):
    return [super(TokenEmbedding, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim)
            ]

  def compute_mask(self, inputs, mask=None):
    return [super(TokenEmbedding, self).compute_mask(inputs, mask), None]

  def call(self, inputs):
    return [super(TokenEmbedding, self).call(inputs), self.embeddings + 0]


class TaskEmbedding(keras.layers.Layer):
  """Embedding for tasks.

      # Arguments
          input_dim: int > 0. Number of the tasks. Plus 1 if `mask_zero` is enabled.
          output_dim: int >= 0. Dimension of the dense embedding.
          embeddings_initializer: Initializer for the `embeddings` matrix.
          embeddings_regularizer: Regularizer function applied to the `embeddings` matrix.
          embeddings_constraint: Constraint function applied to the `embeddings` matrix.
          mask_zero: Generate zeros for 0 index if it is `True`.

      # Input shape
          Previous embeddings, 3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
          Task IDs, 2D tensor with shape: `(batch_size, 1)`.

      # Output shape
          3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
      """

  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               **kwargs):
    super(TaskEmbedding, self).__init__(**kwargs)
    self.support_masking = True
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
    self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
    self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
    self.mask_zero = mask_zero

    self.embeddings = None

  def get_config(self):
    config={
      "input_dim":self.input_dim,
      "output_dim":self.output_dim,
      "embeddings_initializer":
        keras.initializers.serialize(self.embeddings_initializer),
      "embeddings_regularizer":
        keras.regularizers.serialize(self.embeddings_regularizer),
      "embeddings_constraint":
        keras.constraints.serialize(self.embeddings_constraint),
      "mask_zero":self.mask_zero,
    }
    base_config = super(TaskEmbedding,self).get_config()
    return dict(list(base_config.items())+list(config.items()))

  def build(self, input_shape):
    self.embeddings = self.add_weights(
      shape=(self.input_dim, self.output_dim),
      initializer=self.embeddings_initializer,
      regularizer=self.embeddings_regularizer,
      constraint=self.embeddings_constraint,
      name="embeddings",
    )
    super(TokenEmbedding,self).build(input_shape)

  def compute_mask(self, inputs, mask=None):
    output_mask = None
    if mask:
      output_mask = mask[0]
    return output_mask

  def compute_output_shape(self, input_shape):
    return input_shape[0]

  def call(self,inputs, **kwargs):
    inputs, tasks = inputs
    if K.dtype(tasks) != "int32":
      tasks = K.cast(tasks, "int32")
    task_embed = K.gather(self.embeddings, tasks)

    if self.mask_zero:
      task_embed = task_embed * K.expand_dims(
        K.cast(K.not_equal(tasks,0),K.floatx()),
        axis=-1
      )
    return inputs + task_embed

class EmbeddingSimilarity(keras.layers.Layer):
  """Calculate similarity between features and token embeddings with bias term."""
  def __init__(self,
               initializer='zeros',
               regularizer=None,
               constraint=None,
               **kwargs):
    """
    Args:
      initializer: Same as embedding output dimension.
      regularizer: Initializer for bias.
      constraint: Regularizer for bias.
      **kwargs: Constraint for bias.
    """

    super(EmbeddingSimilarity, self).__init__(**kwargs)
    self.support_masking = True
    self.initializer = keras.initializers.get(initializer)
    self.regularizer = keras.regularizers.get(regularizer)
    self.constraint = keras.constraints.get(constraint)

    self.bias = None

  def build(self,input_shape):
    self.bias = self.add_weights(
      shape=(int(input_shape[1][0]),),
      initializer=self.initializer,
      regularizer=self.regularizer,
      constraint=self.constraint,
      name="bias",
    )
    super(EmbeddingSimilarity,self).build(input_shape)

  def call(self, inputs, mask=None, **kwargs):
    inputs, embeddings = inputs
    outputs = K.bias_add(
      K.dot(inputs,K.transpose(embeddings)),
      self.bias)
    return keras.activations.softmax(outputs)

  def compute_mask(self, inputs, mask=None):
    return mask[0]

  def compute_output_shape(self, input_shape):
    return input_shape[0][:2] + (input_shape[1][0], )

  def get_config(self):
    config = {
      "initializer":keras.initializers.serialize(self.initializer),
      "regularizer":keras.regularizers.serialize(self.regularizer),
      "constraint":keras.constraints.serialize(self.constraint),
    }
    base_config = super(EmbeddingSimilarity, self).get_config()
    return dict(list(base_config.items())+list(config.items()))


class Masked(keras.layers.Layer):
  """Generate output mask based on the given mask.
  The inputs for the layer is the original input layer and the masked locations.
  See: https://arxiv.org/pdf/1810.04805.pdf
  """
  def __init__(self,
               return_mask=False,
               **kwargs
               ):
    """
    Args:
      return_mask: Whether to return the merged mask.
      **kwargs:
    """
    super(Masked,self).__init__(**kwargs)
    self.support_masking = True
    self.return_mask = return_mask

  def call(self,inputs, mask=None, **kwargs):
    output = inputs[0] + 0
    if self.return_mask:
      return [output, K.cast(self.compute_mask(inputs, mask)[0], K.floatx())]
    return output

  def compute_mask(self,inputs, mask=None):
    token_mask = K.not_equal(inputs[1], 0)
    masked = K.all(K.stack([token_mask, mask[0]], axis=0), axis=0)
    if self.return_mask:
      return [masked, None]
    return masked

  def compute_output_shape(self, input_shape):
    if self.return_mask:
      return [input_shape[0], input_shape[0][:-1]]
    return input_shape[0]

class Extract(keras.layers.Layer):
  """Extract from index.
  See: https://arxiv.org/pdf/1810.04805.pdf
  """
  def __init__(self,index,**kwargs):
    super(Extract, self).__init__(**kwargs)
    self.support_masking = True
    self.index = index

  def call(self, inputs, mask=None):
    return inputs[:, self.index]

  def compute_mask(self, inputs, mask=None):
    return None

  def compute_output_shape(self, input_shape):
    return input_shape[:1] + input_shape[2:]
