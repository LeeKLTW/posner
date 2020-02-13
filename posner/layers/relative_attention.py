# -*- coding: utf-8 -*-
import tensorflow as tf

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
    bd = rel_shift(bd, klen=tf.shape(ac)[1])

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