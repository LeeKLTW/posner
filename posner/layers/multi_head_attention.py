# -*- coding: utf-8 -*-
"""
.. module:: multi_head_attention
   :synopsis:

.. moduleauthor:: LeeKLTW

"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from . import ScaledDotProductAttention


class MultiHeadAttention(keras.layers.Layer):
  r"""Multi-head attention layer.

  \text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h) W^O
  where head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

  See: https://arxiv.org/pdf/1706.03762.pdf
  """

  def __init__(self,
               head_num,
               activation='relu',
               use_bias=True,
               kernel_initializer='glorot_normal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):
    """Initialize the layer.

    Args:
      head_num: Number of heads.
      activation: Activations for linear mappings.
      use_bias: Whether to use bias term.
      kernel_initializer: Initializer for linear mappings.
      bias_initializer: Initializer for linear mappings.
      kernel_regularizer: Regularizer for linear mappings.
      bias_regularizer: Regularizer for linear mappings.
      kernel_constraint: Constraints for linear mappings.
      bias_constraint: Constraints for linear mappings.
      **kwargs:
    """
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.head_num = head_num
    self.activation = keras.activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = keras.initializers.get(kernel_initializer)
    self.bias_initializer = keras.initializers.get(bias_initializer)
    self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
    self.bias_regularizer = keras.regularizers.get(bias_regularizer)
    self.kernel_constraint = keras.constraints.get(kernel_constraint)
    self.bias_constraint = keras.constraints.get(bias_constraint)

    self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
    self.bq, self.bk, self.bv, self.bo = None, None, None, None


  def get_config(self):
    config = {
      "head_num": self.head_num,
      "activation": keras.activations.serialize(self.activation),
      "use_bias": self.use_bias,
      "kernel_initializer": keras.initializers.serialize(
        self.kernel_initializer),
      "bias_initializer": keras.initializers.serialize(self.bias_initializer),
      "kernel_regularizer": keras.regularizers.serialize(
        self.kernel_regularizer),
      "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
      "kernel_constraint": keras.constraints.serialize(self.kernel_constraint),
      "bias_constraint": keras.constraints.serialize(self.bias_constraint),
    }
    base_config = super(MultiHeadAttention, self).get_config()

    return dict(list(base_config.items()) + list(config.items()))

  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):  # not self-attention case
      q, k, v = input_shape
      return q[:-1] + (v[-1],)
    return input_shape

  def compute_mask(self, inputs, input_mask=None):
    if isinstance(input_mask, list):  # not self-attention case
      return input_mask[0]
    return input_mask

  def build(self, input_shape):
    if isinstance(input_shape, list):  # not self-attention case
      q, k, v = input_shape
    else:
      q = k = v = input_shape

    feature_dim = int(v[-1])
    if feature_dim % self.head_num != 0:
      raise IndexError(
        f"Invalid head number {self.head_num} for given input dim {feature_dim}")

    self.Wq = self.add_weight(
      shape=(int(q[-1]), feature_dim),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      name=f"{self.name}_Wq"
    )
    if self.use_bias:
      self.bq = self.add_weight(
        shape=(feature_dim,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        name=f"{self.name}_bq"
      )

    self.Wk = self.add_weight(
      shape=(int(k[-1]), feature_dim),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      name=f"{self.name}_Wk"
    )
    if self.use_bias:
      self.bk = self.add_weight(
        shape=(feature_dim,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        name=f"{self.name}_bk"
      )

    self.Wv = self.add_weight(
      shape=(int(v[-1]), feature_dim),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      name=f"{self.name}_Wv"
    )
    if self.use_bias:
      self.bv = self.add_weight(
        shape=(feature_dim,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        name=f"{self.name}_bv"
      )

    self.Wo = self.add_weight(
      shape=(feature_dim, feature_dim),
      initializer=self.kernel_initializer,
      regularizer=self.kernel_regularizer,
      constraint=self.kernel_constraint,
      name=f"{self.name}_Wo"
    )

    if self.use_bias:
      self.bo = self.add_weight(
        shape=(feature_dim,),
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        name=f"{self.name}_bo"
      )
    super(MultiHeadAttention, self).build(input_shape)

  def call(self, inputs, mask=None):
    if isinstance(inputs, list):
      q, k, v = inputs
    else:
      q = k = v = inputs

    if isinstance(mask, list):
      q_mask, k_mask, v_mask = mask
    else:
      q_mask = k_mask = v_mask = mask

    q = K.dot(q, self.Wq)
    k = K.dot(k, self.Wk)
    v = K.dot(v, self.Wv)

    if self.use_bias:
      q += self.bq
      k += self.bk
      v += self.bv

    y = ScaledDotProductAttention(name=f"{self.name}-Attention")(
      inputs=[
        self._reshape_to_batches(q, self.head_num),
        self._reshape_to_batches(k, self.head_num),
        self._reshape_to_batches(v, self.head_num),
      ],
      mask=[
        self._reshape_mask(q_mask, self.head_num),
        self._reshape_mask(k_mask, self.head_num),
        self._reshape_mask(v_mask, self.head_num),
      ],
    )

    y = self._reshape_from_batches(y, self.head_num)
    y = K.dot(y, self.Wo)
    if self.use_bias:
      y += self.bo
    if self.activation:
      y = self.activation(y)

    input_shape = [K.int_shape(q), K.int_shape(k), K.int_shape(v)]
    output_shape = self.compute_output_shape(input_shape)
    if output_shape[1] is not None:
      output_shape = (-1,) + output_shape[1:]
      y = K.reshape(y, output_shape)

    return y

  @classmethod
  def _reshape_to_batches(cls, x, head_num):
    input_shape = K.shape(x)
    batch_size, seq_len, feature_dim = input_shape[0],input_shape[1],input_shape[2]
    head_dim = feature_dim // head_num
    x = K.reshape(x,(batch_size, seq_len, head_num, head_dim)) # seperate head_num & head_dim
    x = K.permute_dimensions(x, [0, 2, 1, 3]) # (batch_size, head_num, seq_len, head_dim)
    return K.reshape(x, (batch_size * head_num, seq_len, feature_dim * head_dim))

  @classmethod
  def _reshape_from_batches(cls, x, head_num):
    input_shape = K.shape(x)
    batch_size, seq_len, feature_dim = input_shape[0],input_shape[1],input_shape[2]
    x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
    x = K.permute_dimensions(x, [0,2,1,3])
    return K.reshape(x, (batch_size//head_num, seq_len, feature_dim * head_num))

  @classmethod
  def _reshape_mask(cls, mask, head_num):
    if mask is None:
      return mask
    seq_len = K.shape(mask)[1]
    mask = K.expand_dims(mask,axis=1)
    mask = K.tile(mask, [1, head_num, 1])
    return K.reshape(mask, (-1,seq_len))

#TODO: integrate to attention.py
class RelativeAttention(tf.keras.layers.Layer):
  """Core calculations for relative attention."""

  def __init__(self, dropout_att, scale):
    super(RelativeAttention, self).__init__()
    self.scale = scale
    self.dropout_att = dropout_att

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""

    self.attention_probs_dropout = tf.keras.layers.Dropout(
        rate=self.dropout_att)

    super(RelativeAttention, self).build(unused_input_shapes)

  def call(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
           r_w_bias, r_r_bias, r_s_bias, attn_mask):
    """Implements call() for the layer."""

    # content based attention score
    ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)

    # position based attention score
    bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)
    bd = RelativeAttention.rel_shift(bd, klen=tf.shape(ac)[1])

    # segment-based attention score
    if seg_mat is None:
      ef = 0
    else:
      ef = tf.einsum('ibnd,snd->isbn', q_head + r_s_bias, seg_embed)
      tgt_shape = tf.shape(bd)
      ef = tf.where(
          tf.broadcast_to(tf.expand_dims(seg_mat, 3), tgt_shape),
          tf.broadcast_to(ef[:, 1:, :, :], tgt_shape),
          tf.broadcast_to(ef[:, :1, :, :], tgt_shape))

    # merges attention scores and performs masking
    attn_score = (ac + bd + ef) * self.scale
    if attn_mask is not None:
      attn_score = attn_score - 1e30 * attn_mask

    # attention probability
    attn_prob = tf.nn.softmax(attn_score, 1)
    attn_prob = self.attention_probs_dropout(attn_prob)

    # attention output
    attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

    return attn_vec

  @staticmethod
  def rel_shift(x, klen=-1):
    """Performs relative shift to form the relative attention score."""
    x_size = tf.shape(x)

    x = tf.reshape(x, [x_size[1], x_size[0], x_size[2], x_size[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [x_size[0], x_size[1] - 1, x_size[2], x_size[3]])
    x = tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

    return x

class RelativeMultiheadAttention(tf.keras.layers.Layer):
  """Multi-head attention with relative embedding."""

  def __init__(self, d_model, n_head, d_head, dropout, dropout_att,
               kernel_initializer, **kwargs):
    super(RelativeMultiheadAttention, self).__init__(**kwargs)
    self.d_model = d_model
    self.n_head = n_head
    self.d_head = d_head
    self.dropout = dropout
    self.dropout_att = dropout_att
    self.initializer = kernel_initializer

  def build(self, unused_input_shapes):
    """Implements build() for the layer."""
    self.scale = 1.0 / (self.d_head ** 0.5)

    self.output_layer_norm = tf.keras.layers.LayerNormalization(
      name='LayerNorm', axis=-1, epsilon=1e-12)

    self.kh_projection_layer = self.add_weight(
      'k/kernel',
      shape=[self.d_model, self.n_head, self.d_head],
      initializer=self.initializer)
    self.vh_projection_layer = self.add_weight(
      'v/kernel',
      shape=[self.d_model, self.n_head, self.d_head],
      initializer=self.initializer)
    self.kr_projection_layer = self.add_weight(
      'r/kernel',
      shape=[self.d_model, self.n_head, self.d_head],
      initializer=self.initializer)
    self.qh_projection_layer = self.add_weight(
      'q/kernel',
      shape=[self.d_model, self.n_head, self.d_head],
      initializer=self.initializer)

    self.relative_attention_layer = RelativeAttention(
      dropout_att=self.dropout_att, scale=self.scale)

    self.proj_o = self.add_weight(
      'o/kernel',
      shape=[self.d_model, self.n_head, self.d_head],
      initializer=self.initializer)

    self.attention_dropout = tf.keras.layers.Dropout(rate=self.dropout)

    super(RelativeMultiheadAttention, self).build(unused_input_shapes)

  def call(self, h, g, r, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
           attn_mask_h, attn_mask_g, mems, target_mapping):
    """Implements call() for the layer."""

    if mems is not None and mems.shape.ndims > 1:
      cat = tf.concat([mems, h], 0)
    else:
      cat = h

    # content heads
    q_head_h = tf.einsum('ibh,hnd->ibnd', h, self.qh_projection_layer)
    k_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.kh_projection_layer)
    v_head_h = tf.einsum('ibh,hnd->ibnd', cat, self.vh_projection_layer)

    # positional heads
    k_head_r = tf.einsum('ibh,hnd->ibnd', r, self.kr_projection_layer)

    # core attention ops
    attn_vec_h = self.relative_attention_layer(q_head_h, k_head_h, v_head_h,
                                               k_head_r, seg_embed, seg_mat,
                                               r_w_bias, r_r_bias, r_s_bias,
                                               attn_mask_h)

    # post processing
    output_h = tf.einsum('ibnd,hnd->ibh', attn_vec_h, self.proj_o)
    output_h = self.attention_dropout(output_h)
    output_h = self.output_layer_norm(output_h + h)

    output_g = None
    if g is not None:  # enable two-stream attention
      # g-stream
      q_head_g = tf.einsum('ibh,hnd->ibnd', g, self.qh_projection_layer)
      if target_mapping is not None:
        q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
        attn_vec_g = self.relative_attention_layer(q_head_g, k_head_h, v_head_h,
                                                   k_head_r, seg_embed, seg_mat,
                                                   r_w_bias, r_r_bias, r_s_bias,
                                                   attn_mask_g)
        attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)

      else:
        attn_vec_g = self.relative_attention_layer(q_head_g, k_head_h, v_head_h,
                                                   k_head_r, seg_embed, seg_mat,
                                                   r_w_bias, r_r_bias, r_s_bias,
                                                   attn_mask_g)

      # post processing
      output_g = tf.einsum('ibnd,hnd->ibh', attn_vec_g, self.proj_o)
      output_g = self.attention_dropout(output_g)
      output_g = self.output_layer_norm(output_g + g)

    return (output_h, output_g)
