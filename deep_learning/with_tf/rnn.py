import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)

learning_rate = 0.001
training_iters = 100000
batch_size = 128
display = 10

img_size = 28
n_steps = 28
n_features = 128
n_classes = 10

x = tf.placeholder("float32", [None, n_steps, img_size])
y = tf.placeholder("float32", [None, n_classes])

weights = tf.Variable(tf.random_normal([n_features, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))


def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    lstm_cell = rnn.BasicLSTMCell(n_features, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights) + biases


y_pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(y_pred, dimension=1), tf.argmax(y, dimension=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, n_steps, img_size))
        sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})
        if step % display == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict={x: batch_x, y:batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

test_size = 128
test_data = mnist.test.images[:test_size].reshape((-1, n_steps, img_size))
test_labels = mnist.test.labels[:test_size]
print("Testing Accuracy: %s" % sess.run(accuracy, feed_dict={x: test_data, y: test_labels}))


