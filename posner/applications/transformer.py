# -*- coding: utf-8 -*-
from tensorflow import keras

from posner.layers import LayerNormalization, MultiHeadAttention, \
  PositionFeedForward, PositionEmbedding


def get_custom_objects():
  return {
    "LayerNormalization": LayerNormalization,
    "MultiHeadAttention": MultiHeadAttention,
    "PositionFeedForward": PositionFeedForward,
    "PositionEmbedding": PositionEmbedding,
  }


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True,
                use_adapter=False,
                adapter_units=None,
                adapter_activation='relu'):
  """Wrap layers with residual, normalization and dropout.

  Args:
    name: Prefix of names for internal layers.
    input_layer:  Input layer.
    build_func: A callable that takes the input tensor and generates the output tensor.
    dropout_rate: Dropout rate.
    trainable: Whether the layers are trainable.
    use_adapter:  Whether to use feed-forward adapters before each residual connections.
    adapter_units: The dimension of the first transformation in feed-forward adapter.
    adapter_activation: The activation after the first transformation in feed-forward adapter.

  Returns:
    Output layer.

  """
  build_output = build_func(input_layer)
  if dropout_rate > 0.0:
    dropout_layer = keras.layers.Dropout(rate=dropout_rate,
                                         name=f"{name}-Dropout")(build_output)
  else:
    dropout_layer = build_output

  if isinstance(input_layer, list):
    input_layer = input_layer[0]

  if use_adapter:
    adapter = PositionFeedForward(
      units=adapter_units,
      activation=adapter_activation,
      kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
      name=f"{name}-Adapter"
    )(dropout_layer)
    dropout_layer = keras.layers.Add(name=f"{name}-Adapter-Add")([dropout_layer, adapter])
  add_layer = keras.layers.Add(name=f"{name}-Add")([input_layer, dropout_layer])
  normal_layer = LayerNormalization(trainable=trainable,
                                    name=f"{name}-Norm",
  )(add_layer)
  return normal_layer


def attention_builder(name,
                      head_num,
                      activation,
                      trainable=True):
  """ Get multi-head self-attention builder.

  Args:
    name: Prefix of names for internal layers.
    head_num: Number of heads in multi-head self-attention.
    activation: Activation for multi-head self-attention.
    trainable: Whether the layer is trainable.

  Returns:
    builder function
  """
  def _attention_builder(x):
    return MultiHeadAttention(head_num=head_num,
                              activation=activation,
                              trainable=trainable,
                              name=name
                              )(x)

  return _attention_builder


def feed_forward_builder(name,
                         hidden_dim,
                         activation,
                         trainable=True):
  """Get position-wise feed-forward layer builder.

  Args:
    name: Prefix of names for internal layers.
    hidden_dim: Hidden dimension of feed forward layer.
    activation: Activation for feed-forward layer.
    trainable: Whether the layer is trainable.

  Returns:
    builder function

  """
  def _feed_forward_builder(x):
    return PositionFeedForward(units=hidden_dim,
                               activation=activation,
                               trainable=trainable,
                               name=name
                               )(x)

  return _feed_forward_builder



def get_encoder_componet(name,
                        input_layer,
                        head_num,
                        hidden_dim,
                        attention_activation=None,
                        feed_forward_activation='relu',
                        dropout_rate=0.0,
                        trainable=True,
                        use_adapter=False,
                        adapter_units=None,
                        adapter_activation='relu'):
  """

  Args:
    name: Prefix of names for internal layers.
    input_layer: Input layer.
    head_num: Number of heads in multi-head self-attention.
    hidden_dim: Hidden dimension of feed forward layer.
    attention_activation: Activation for multi-head self-attention.
    feed_forward_activation: Activation for feed-forward layer.
    dropout_rate: Dropout rate.
    trainable:  Whether the layers are trainable.
    use_adapter: Whether to use feed-forward adapters before each residual connections.
    adapter_units: The dimension of the first transformation in feed-forward adapter.
    adapter_activation: The activation after the first transformation in feed-forward adapter.

  Returns:
    Output layer.

  """
  attention_name = f"{name}-MultiHeadSelfAttention"
  feed_forward_name = f"{name}-FeedForward"
  attention_layer = _wrap_layer(
    name=attention_name,
    input_layer=input_layer,
    build_func=attention_builder(
      name=attention_name,
      head_num=head_num,
      activation=attention_activation,
      trainable=trainable,
    ),
    dropout_rate=dropout_rate,
    trainable=trainable,
    use_adapter=use_adapter,
    adapter_units=adapter_units,
    adapter_activation=adapter_activation
  )

  feed_forward_layer = _wrap_layer(
    name=feed_forward_name,
    input_layer=attention_layer,
    build_func=feed_forward_builder(
      name=feed_forward_name,
      hidden_dim=hidden_dim,
      activation=feed_forward_activation,
      trainable=trainable
    ),
    dropout_rate=dropout_rate,
    trainable=trainable,
    use_adapter=use_adapter,
    adapter_units=adapter_units,
    adapter_activation=adapter_activation
  )

  return feed_forward_layer




def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',):
  """ Get encoders.

  Args:
    encoder_num: Number of encoder components
    input_layer: Input layer.
    head_num: Number of heads in multi-head self-attention.
    hidden_dim: Hidden dimension of feed forward layer.
    attention_activation: Activation for multi-head self-attention.
    feed_forward_activation: Activation for feed-forward layer.
    dropout_rate: Dropout rate.
    trainable: Whether the layers are trainable.
    use_adapter: Whether to use feed-forward adapters before each residual connections.
    adapter_units: The dimension of the first transformation in feed-forward adapter.
    adapter_activation: The activation after the first transformation in feed-forward adapter.

  Returns:
    Output layer.

  """
  last_layer = input_layer
  for i in range(encoder_num):
    last_layer = get_encoder_componet(
      name=f"Encoder-{i+1}",
      input_layer=last_layer,
      head_num=head_num,
      hidden_dim=hidden_dim,
      attention_activation=attention_activation,
      feed_forward_activation=feed_forward_activation,
      dropout_rate=dropout_rate,
      trainable=trainable,
      use_adapter=use_adapter,
      adapter_units=adapter_units,
      adapter_activation=adapter_activation,
    )
  return last_layer


def get_decoders(decoder_num,
                 input_layer,
                encoded_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation="relu",
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation="relu",):
  """ Get decoders.

  Args:
    decoder_num: Number of decoder components.
    input_layer: Input layer.
    encoded_layer: Encoded layer from encoder.
    head_num: Number of heads in multi-head self-attention.
    hidden_dim: Hidden dimension of feed forward layer.
    attention_activation: Activation for multi-head self-attention.
    feed_forward_activation: Activation for feed-forward layer.
    dropout_rate: Dropout rate.
    trainable: Whether the layers are trainable.
    use_adapter: Whether to use feed-forward adapters before each residual connections.
    adapter_units: The dimension of the first transformation in feed-forward adapter.
    adapter_activation: The activation after the first transformation in feed-forward adapter.

  Returns:
    Output layer.
  """
  last_layer = input_layer
  for i in range(decoder_num):
    last_layer = get_decoder_component()
  return last_layer