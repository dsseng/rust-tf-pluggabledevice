# from https://github.com/tensorflow/community/pull/352

import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()

a = tf.random.normal(shape=[1, 28, 28, 8], dtype=tf.float32, seed=1)
w = tf.random.normal(shape=[3, 3, 8, 4], dtype=tf.float32, seed=1)

a_p = tf.random.normal(shape=[1, 28, 28, 8], dtype=tf.float32, seed=1)
w_p = tf.random.normal(shape=[3, 3, 8, 4], dtype=tf.float32, seed=1)

# PluggableDevice's have top priority, so TF will accelerate what's possible
b_p = tf.nn.relu(a_p)
c_p = tf.nn.conv2d(b_p, w_p, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

with tf.device("/CPU:0"):
    b = tf.nn.relu(a)
    c = tf.nn.conv2d(b, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NHWC')

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=True))

print(sess.run(tf.reduce_all(tf.less(c_p - c, 1e-5))))
