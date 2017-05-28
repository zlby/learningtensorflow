from serialize_data import SequenceStockData
from get_data import get_data_list
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from model import Model

data_size = 1000
learning_rate = 0.0001
display_step = 10
training_epochs = 1000


x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

weights = {
    "out": tf.Variable(np.random.randn())
}
biases = {
    "out": tf.Variable(np.random.randn())
}

pred = tf.add(tf.multiply(x, weights["out"]), biases["out"])

cost = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * data_size)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

file_name = 'test.csv'
data_list = get_data_list(file_name)
train_set = SequenceStockData(data_list, 0, data_size)
train_x, train_y = train_set.next_batch(data_size)

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		for (xs, ys) in zip(train_x, train_y):
			sess.run(optimizer, feed_dict={x: xs, y: ys})

		if (epoch + 1) % display_step == 0:
			c = sess.run(cost, feed_dict={x: xs, y: ys})
			print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))