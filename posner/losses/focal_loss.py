# -*- coding: utf-8 -*-
import tensorflow as tf
from keras import keras
from tensorflow.keras import backend as K


class BinaryFocalLoss(keras.losses.Loss):
  """
  Binary form of focal loss.
    FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
    where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
  References:
      https://arxiv.org/pdf/1708.02002.pdf
  """

  def __init__(self, gamma=2., alpha=0.25):
    self.gamma = gamma
    self.alpha = alpha

  def call(self, y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()

    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    loss = -K.sum(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1)) \
           - K.sum(
      (1 - self.alpha) * K.pow(pt_0, self.gamma) * K.log(1. - pt_0))
    return loss

  def get_config(self):
    config = {"gamma": self.gamma,
              "alpha": self.alpha}
    base_config = super(BinaryFocalLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class CategoricalFocalLoss(keras.losses.Loss):
  """
  Softmax version focal loss for multi category of multi label problem
             m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation

  """
  def __init__(self, gamma=2., alpha=.25):
    self.gamma = gamma
    self.alpha = alpha

  def call(self, y_true, y_pred):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    cross_entropy = -y_true * K.log(y_pred)

    loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy

    return K.sum(loss, axis=1)

  def get_config(self):
    config = {"gamma": self.gamma,
              "alpha": self.alpha}
    base_config = super(CategoricalFocalLoss, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
