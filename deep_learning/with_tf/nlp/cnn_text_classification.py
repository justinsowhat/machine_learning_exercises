import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
from deep_learning.with_tf.nlp import data_helpers


class CNN(object):
    def __init__(self, seq_len, num_classes, vocab_size, embedding_size, filter_sizes, num_filters, l2_reg=0.0):

        self.X = tf.placeholder(tf.int32, [None, seq_len])
        self.y = tf.placeholder(tf.float, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        l2_loss = tf.constant(0.0)

        self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.X)
        self.embedded_chars_expaneded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for filter_size in filter_sizes:
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(
                self.embedded_chars_expaneded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h,
                                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding='VALID')
            pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        self.h_dropout = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        self.scores = tf.nn.xw_plus_b(self.h_dropout, W, b)
        self.predictions = tf.argmax(self.scores, dimension=1)

        losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y)
        self.loss = tf.reduce_mean(losses) + l2_reg * l2_loss

        correct_prediction = tf.equal(self.predictions, tf.argmax(self.y, dimension=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


POS_TEXT = 'data/rt-polarity.pos'
NEG_TEXT = 'data/rt-polarity.neg'

x_text, y = data_helpers.load_data_and_labels(POS_TEXT, NEG_TEXT)

max_doc_len = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_doc_len)
x = np.array(list(vocab_processor.fit_transform(x_text)))

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

dev_index = -1 * int(0.1 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_index], x_shuffled[dev_index:]
y_train, y_dev = y_shuffled[:dev_index], y_shuffled[dev_index:]


g = tf.Graph()

with g.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    session = tf.Session(config=session_conf)
    with session.as_default():
        cnn = CNN(
            seq_len=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=100,
            filter_sizes=[3, 4, 5],
            l2_reg=0.0
        )

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    session.run(tf.global_variables_initializer())

    # TODO
    # feed in data in train and dev steps...
    # gradient descent...
    # evaluation...
