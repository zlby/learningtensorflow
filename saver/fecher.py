import tensorflow as tf
import numpy as np

W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name = 'weights')
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name = 'biases')

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, "save/model.ckpt")
	print("weight: ", sess.run(W))
	print("biases: ", sess.run(b))