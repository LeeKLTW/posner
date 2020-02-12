# -*- coding: utf-8 -*-
import tensorflow as tf

def gelu(input_tensor):
  cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / tf.sqrt(2.0)))
  return input_tensor * cdf
