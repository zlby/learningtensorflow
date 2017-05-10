from serialize_data import SequenceStockData
from get_data import get_data_list
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

learning_rate = 0.001
training_iters = 1000000
batch_size = 10
display_step = 100

train_start = 0
train_end = 1000
test_start = 0
test_end = 1000

n_step = 1
n_input = 1
n_hidden = 128
n_class = 1

x = tf.placeholder(tf.float32, [None, n_step, n_input])
y = tf.placeholder(tf.float32, [None, n_class])

weights = {
    "out": tf.Variable(tf.random_normal([2 * n_hidden, n_class]))
}
biases = {
    "out": tf.Variable(tf.random_normal([n_class]))
}


def Model(x, weights, biases):
    x = tf.unstack(x, n_step, 1)

    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    outputs, _, _ = rnn.static_bidirectional_rnn(
        lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    return tf.add(tf.matmul(outputs[-1], weights["out"]), biases["out"])

pred = Model(x, weights, biases)

cost = tf.reduce_sum(tf.pow(pred - y, 2)) / (2 * batch_size)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# correct_num = 0
# for i in range(len(pred)):
# 	if abs(pred[i] - y[i]) < 0.0001:
# 		correct_num += 1
# accuracy = tf.div(correct_num / len(pred))

init = tf.global_variables_initializer()

file_name = 'test.csv'
data_list = get_data_list(file_name)
train_set = SequenceStockData(data_list, train_start, train_end)
test_set = SequenceStockData(data_list, test_start, test_end)

with tf.Session() as sess:
	sess.run(init)
	step = 1
	while step * batch_size < training_iters:
		batch_x, batch_y = train_set.next_batch(batch_size)
		batch_x = np.array(batch_x)
		batch_x = batch_x.reshape((batch_size, n_step, n_input))
		batch_y = np.array(batch_y)
		batch_y = batch_y.reshape((batch_size, n_class))
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		if step % display_step == 0:
			# acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
			loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
			print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss))
		step += 1
	print("Optimization Finished!")