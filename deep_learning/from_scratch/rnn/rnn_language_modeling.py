import csv
import itertools
import numpy as np
import nltk

# nltk.download("book")

vocabulary_size = 8000
unknown_token = "UNKNOWN"
sentence_start_token = "<S>"
sentence_end_token = "</S>"

print("Reading CSV file...")
with open('deep_learning/from_scratch/rnn/data/reddit-comments-2015-08.csv', 'r') as f:
    reader = csv.reader(f, skipinitialspace=True)
    f.readline()
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # padding sentences
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print("Found %d unique words tokens." % len(word_freq.items()))

# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print("Using vocabulary size %d." % vocabulary_size)

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]


hidden_size = 100
sequence_len = 4
learning_rate = 0.001

# input to hidden
Wxh = np.random.randn(hidden_size, vocabulary_size) * 0.01
# hidden to hidden
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
# hidden to output
Why = np.random.randn(vocabulary_size, hidden_size) * 0.01
# bias for the hidden layer
bh = np.zeros((hidden_size, 1))
# bias for the output layer
by = np.zeros((vocabulary_size, 1))


def loss(inputs, targets, prev_hidden):
    xs, hs, ys, probs = {}, {}, {}, {}
    hs[-1] = np.copy(prev_hidden)
    loss = 0
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocabulary_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        probs[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(probs[t][targets[t], 0])

    # partial derivatives
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    # calculating the partial derivatives
    for t in reversed((range(len(inputs)))):
        dy = np.copy(probs[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext

        dhraw = (1 - hs[t] * hs[t]) * dh

        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs) - 1]


def sample(state, seed_word_idx, num_words):
    x = np.zeros((vocabulary_size, 1))
    x[seed_word_idx] = 1
    idxes = []

    for t in range(num_words):
        state = np.tanh(np.dot(Wxh, x) + np.dot(Whh, state) + bh)
        y = np.dot(Why, state) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ix = np.random.choices(range(vocabulary_size), p=p.ravel())
        x = np.zeros((vocabulary_size, 1))
        x[ix] = 1
        idxes.append(ix)

    sent = ' '.join(index_to_word[idx] for idx in idxes)
    print(sent)


itr, data_idx = 0, 0

# memory variables
memWxh = np.zeros_like(Wxh)
memWhh = np.zeros_like(Whh)
memWhy = np.zeros_like(Why)
membh = np.zeros_like(bh)
memby = np.zeros_like(by)

smooth_loss = -np.log(1.0 / vocabulary_size) * sequence_len

while itr <= 100000:
    if data_idx + sequence_len + 1 >= len(tokenized_sentences) or itr == 0:
        prev_hidden = np.zeros((hidden_size, 1))
        p = 0

    # Create the inputs
    X = np.asarray([[word_to_index[w] for w in sent[data_idx:data_idx+sequence_len]]
                    for sent in tokenized_sentences])
    # the target shifts one compared to the input
    y = np.asarray([[word_to_index[w] for w in sent[data_idx+1:data_idx+sequence_len+1]]
                    for sent in tokenized_sentences])

    loss, dWxh, dWhh, dWhy, dbh, dby, prev_hidden = loss(X, y, prev_hidden)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    if itr % 10 == 0:
        print("iteration %d, loss: %f" % (itr, smooth_loss))
        sample(prev_hidden, X[0], 10)

    # adagrad update
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [memWxh, memWhh, memWhy, membh, memby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)

    data_idx += sequence_len
    itr += 1

