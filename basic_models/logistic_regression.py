import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y *
                                              tf.log(pred), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={
                            x: batch_xs, y: batch_ys})
            avg_loss += c / total_batch

        if (epoch + 1) % display_step == 0:
            print("Epoch:", epoch + 1, "loss=", "{:.9f}".format(avg_loss))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy:", accuracy.eval(
        {x: mnist.test.images, y: mnist.test.labels}))
