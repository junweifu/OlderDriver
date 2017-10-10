# -*- coding: utf-8 -*-

import tensorflow as tf

def add_layer(inputs,in_size,out_size,activation_function=None):
    w = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size])+0.1)
    y = tf.matmul(inputs,w)+b
    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs