# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def add_layer(inputs,input_size,output_size,activation_function=None):
    w = tf.Variable(tf.random_normal([input_size,output_size]))
    b = tf.Variable(tf.zeros([1,output_size])+0.1)
    y = tf.matmul(inputs,w)+b
    if activation_function is None:
        outputs = y
    else:
        outputs = activation_function(y)
    return outputs

x_data = np.linspace(-1,1,300,dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

xs = tf.placeholder(tf.float32,[None,1],name='x_input')
ys = tf.placeholder(tf.float32,[None,1],name='y_input')
    
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)

prediction = add_layer(l1,10,1,activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                    reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
