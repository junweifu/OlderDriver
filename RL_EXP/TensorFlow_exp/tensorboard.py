# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def add_layer(inputs,input_size,output_size, n_layer, activation_function=None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            w = tf.Variable(tf.random_normal([input_size,output_size]),name='W')
            tf.summary.histogram(layer_name+'/w',w)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([1,output_size])+0.1,name='b')
            tf.summary.histogram(layer_name+'/b',b)
        with tf.name_scope('y'):
            y = tf.add(tf.matmul(inputs,w),b)
        if activation_function is None:
            outputs = y
        else:
            outputs = activation_function(y)
        tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

x_data = np.linspace(-1,1,300,dtype = np.float32)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
    
l1 = add_layer(xs,1,10,n_layer = 1,activation_function=tf.nn.relu)

prediction = add_layer(l1,10,1,n_layer = 2,activation_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),
                                        reduction_indices=[1]))
    tf.summary.scalar('loss', loss)
    
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i % 50 == 0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        result = sess.run(merged,
                          feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)