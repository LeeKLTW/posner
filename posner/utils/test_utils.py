# -*- coding: utf-8 -*-
import tensorflow as tf


def to_tuple(shape):
  return tuple(tf.TensorShape(shape).as_list())