import numpy as np
import tensorflow as tf

def create_model(images, batch_size, n_classes):
    # conv1 layer
    with tf.name_scope("conv1") as scope:
        weights = tf.get_variable("weight", shape=[5, 5, 3, 62], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[62], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1 layer
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2 layer
    with tf.name_scope("conv2") as scope:
        weights = tf.get_variable("weight", shape=[5, 5, 62, 62*62], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[62*62], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # pool2 layer
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

    # conv3 layer
    with tf.name_scope("conv3") as scope:
        weights = tf.get_variable("weight", shape=[5, 5, 62*62, 62*62*62], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[62*62*62], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)

        # pool3 layer
    with tf.variable_scope('pooling3_lrn') as scope:
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling3')
        norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

        # conv4 layer
    with tf.name_scope("conv4") as scope:
        weights = tf.get_variable("weight", shape=[5, 5, 62*62*62, 62 * 62*62*62], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases", shape=[62 * 62*62*62], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm3, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)

        # pool2 layer
    with tf.variable_scope('pooling4_lrn') as scope:
        norm3 = tf.nn.lrn(conv4, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')
        pool4 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling4')


