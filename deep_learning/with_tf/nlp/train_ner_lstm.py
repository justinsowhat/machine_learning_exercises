# tensorflow == 0.9.0rc0
# tflearn == 0.2.1
# this code references the implementation here: https://github.com/dhwajraj/NER-RNN

# this was a course project to train a Twitter NER tagger with LSTM

import tensorflow as tf
from tensorflow.python.framework import ops
import tflearn
from tensorflow.python.ops.rnn_cell import MultiRNNCell, GRUCell
from tensorflow.python.ops import rnn
import numpy as np
import re
import gc


TRAIN_PATH = './data/train.pos.gold'
DEV_PATH = './data/dev.pos.gold'
TEST_PATH = './data/test.pos.gold'
# Twitter Brown Cluster: http://www.cs.cmu.edu/~ark/TweetNLP/clusters/50mpaths2
BROWN_PATH = './50mpaths2.txt'
# Twitter word2vec model: http://www.fredericgodin.com/software/
W2V = './twitter_w2v_subset.txt'
BATCH_SIZE = 50
RESTORE = None
CLASS_SIZE = 3
EPOCH = 1


def index_brown(file):
    indices = []
    words = dict()
    with open(file, encoding='utf8') as f:
        for line in f:
            try:
                cluster, w, c = line.split('\t')
                if cluster not in indices:
                    indices.append(cluster)
                words[w] = indices.index(cluster)
            except:
                pass
    return words


def pos(tag):
    POS_TAGS = ['N', 'O', 'S', '^', 'Z', 'L', 'M', 'V', 'A', 'R', '!', 'D', 'P', '&', 'T', 'X', 'Y', '#', '@', '~',
                'U', 'E', '$', ',', 'G']
    onehot = np.zeros(25)
    onehot[POS_TAGS.index(tag)] = 1
    return onehot


def is_upper(word):
    if word.isupper:
        return np.asarray([1])
    else:
        return np.asarray([0])


def is_digit(word):
    if word.isdigit:
        return np.asarray([1])
    else:
        return np.asarray([0])


def brown(w, indices, size):
    onehot = np.zeros(size)
    if w in indices:
        onehot[indices[w]] = 1
    return onehot


def has_hashtag(word):
    if word.startswith('#'):
        return np.asarray([1])
    else:
        return np.asarray([0])


def has_at(word):
    if word.startswith('@'):
        return np.asarray([1])
    else:
        return np.asarray([0])


def load_w2v(path):
    d = dict()
    for line in open(path):
        l = line.strip().split()
        w = l[0]
        d[w] = np.asarray(l[1:], dtype='float32')
    return d


def initialize_values(path):
    # finds the max num of words in sentences
    # and indexes the words
    max_sent_len = 0
    word_indices = {}
    with open(path) as f:
        i = 0
        for line in f:
            if len(line.strip()) == 0:
                max_sent_len = max(i, max_sent_len)
                i = 0
            else:
                w = line.split('\t')[0].lower()
                word_indices[w] = len(word_indices)
                i += 1
    return (max_sent_len + 1), word_indices

print("Indexing Brown Clusters....")
BROWN_CLUSTER = index_brown(BROWN_PATH)
CLUSTER_SIZE = len(set(BROWN_CLUSTER.values()))
FEATURE_SIZE = 27 + CLUSTER_SIZE
WORD_VECTOR_SIZE = 400
print("Indexing word2vec vectors...")
WORD_EMBEDDING = load_w2v(W2V)
MAX_DOCUMENT_LENGTH, WORD_INDICES = initialize_values(TRAIN_PATH)
VOCAB_SIZE = 0  # len(WORD_INDICES)
EMBEDDING_SIZE = WORD_VECTOR_SIZE + FEATURE_SIZE + VOCAB_SIZE


def get_word_embedding(w):
    randV = np.random.uniform(-0.25, 0.25, WORD_VECTOR_SIZE)
    symbol = re.sub('[^0-9a-zA-Z]+', '', w)
    arr = []
    if w in WORD_EMBEDDING:
        arr = WORD_EMBEDDING[w]
    elif w.lower() in WORD_EMBEDDING:
        arr = WORD_EMBEDDING[w.lower()]
    elif symbol in WORD_EMBEDDING:
        arr = WORD_EMBEDDING[symbol]
    if len(arr) > 0:
        return np.asarray(arr)
    return randV


def get_word_indices(w):
    w = w.lower()
    randV = np.random.uniform(-0.25, 0.25, VOCAB_SIZE)
    if w in WORD_INDICES:
        onehot = np.zeros(VOCAB_SIZE)
        onehot[WORD_INDICES[w]] = 1
        return onehot
    else:
        return randV


def parse_file(path):
    word = []
    tag = []

    sent = []
    sent_tag = []

    max_sent_len = MAX_DOCUMENT_LENGTH
    sent_len = 0

    for line in open(path):
        if line in ['\n', '\r\n']:
            for _ in range(max_sent_len - sent_len):
                tag.append(np.asarray([0, 0, 0]))
                temp = np.array([0 for _ in range(EMBEDDING_SIZE)])
                word.append(temp)

            sent.append(word)
            sent_tag.append(np.asarray(tag))
            sent_len = 0
            word = []
            tag = []
        else:
            assert (len(line.split('\t')) == 3)
            sent_len += 1
            # get word, pos, and tag for the line
            w, p, t = line.strip().split('\t')
            # temp = get_word_indices(w)
            # temp = np.append(temp, get_word_embedding(w))
            temp = get_word_embedding(w)
            temp = np.append(temp, pos(p))
            temp = np.append(temp, is_upper(w))
            temp = np.append(temp, is_digit(w))
            # temp = np.append(temp, has_hashtag(w))
            # temp = np.append(temp, has_at(w))
            temp = np.append(temp, brown(w, BROWN_CLUSTER, CLUSTER_SIZE))
            word.append(temp)
            if t == 'O':
                tag.append(np.asarray([1, 0, 0]))
            elif t == 'B':
                tag.append(np.asarray([0, 1, 0]))
            elif t == 'I':
                tag.append(np.asarray([0, 0, 1]))
            else:
                print("wrong tag %s" % t)

    assert (len(sent) == len(sent_tag))
    return np.asarray(sent), sent_tag


print('Parsing files...')
X_train, y_train = parse_file(TRAIN_PATH)
X_dev, y_dev = parse_file(DEV_PATH)
X_test, y_test = parse_file(TEST_PATH)

y_train = np.asarray(y_train).astype(int).reshape(len(y_train), -1)
y_dev = np.asarray(y_dev).astype(int).reshape(len(y_dev), -1)
y_test = np.asarray(y_test).astype(int).reshape(len(y_test), -1)

del WORD_EMBEDDING
del WORD_INDICES
del BROWN_CLUSTER


def cost(prediction, target):
    target = tf.reshape(target, [-1, MAX_DOCUMENT_LENGTH, CLASS_SIZE])
    prediction = tf.reshape(prediction, [-1, MAX_DOCUMENT_LENGTH, CLASS_SIZE])
    cross_entropy = target * tf.log(prediction)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    cross_entropy *= mask
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.cast(length(target), tf.float32)
    return tf.reduce_mean(cross_entropy)


def length(target):
    used = tf.sign(tf.reduce_max(tf.abs(target), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def f1(prediction, target):  # not tensors but result values
    target = np.reshape(target, (-1, MAX_DOCUMENT_LENGTH, CLASS_SIZE))
    prediction = np.reshape(prediction, (-1, MAX_DOCUMENT_LENGTH, CLASS_SIZE))

    tp = np.asarray([0] * (CLASS_SIZE + 2))
    fp = np.asarray([0] * (CLASS_SIZE + 2))
    fn = np.asarray([0] * (CLASS_SIZE + 2))

    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)

    for i in range(len(target)):
        for j in range(MAX_DOCUMENT_LENGTH):
            if target[i][j] == prediction[i][j]:
                tp[target[i][j]] += 1
            else:
                fp[target[i][j]] += 1
                fn[prediction[i][j]] += 1

    NON_NAMED_ENTITY = 0
    for i in range(CLASS_SIZE):
        if i != NON_NAMED_ENTITY:
            tp[3] += tp[i]
            fp[3] += fp[i]
            fn[3] += fn[i]
        else:
            tp[4] += tp[i]
            fp[4] += fp[i]
            fn[4] += fn[i]

    precision = []
    recall = []
    fscore = []
    for i in range(CLASS_SIZE + 2):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))

    print("precision = ", precision)
    print("recall = ", recall)
    print("f1score = ", fscore)
    efs = fscore[4]
    print("Entity fscore :", efs)
    del precision
    del recall
    del fscore
    return efs

print("Network building...")

net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE])
net = rnn.bidirectional_rnn(MultiRNNCell([GRUCell(256)] * 3), MultiRNNCell([GRUCell(256)] * 3),
                            tf.unpack(tf.transpose(net, perm=[1, 0, 2])),
                            dtype=tf.float32)  # 256=num_hidden, 3=num_layers
net = tflearn.dropout(net[0], 0.5)
net = tf.transpose(tf.pack(net), perm=[1, 0, 2])

net = tflearn.fully_connected(net, MAX_DOCUMENT_LENGTH * CLASS_SIZE, activation='softmax')
net = tflearn.regression(net, optimizer='adam', loss=cost)

model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=0)

print("Training...")

gc.collect()
while True:
    with ops.get_default_graph().as_default():
        model.fit(X_train, y_train, n_epoch=EPOCH, validation_set=(X_dev, y_dev), show_metric=False, batch_size=BATCH_SIZE)
        y_pred = np.asarray(model.predict(X_test))
        print("Performance on the test set:")
        f1(y_pred, y_test)
        del y_pred
        gc.collect()


