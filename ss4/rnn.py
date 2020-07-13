from collections import defaultdict
from os import listdir
from os.path import isfile
import re
import tensorflow as tf
import numpy as np

# Mạng nơ-ron hồi quy - Recurrent Neural Networks ra đời.
# Chuyên dùng cho dữ liệu dạng chuỗi.
# Mô hình hóa quan hệ thứ tự giữa các phần tử trong chuỗi
# mô hình hoá ngôn , dịch máy, nhận dạng giọng nói, mô tả hình ảnh

tf.disable_eager_execution()

MAX_DOC_LENGTH = 500
NUM_CLASSES = 4
unknown_ID = 1
padding_ID = 0

def gen_data_and_vocab():
  def collect_data_from(parrent_path, newsgroup_list, word_count=None):
    data = []
    for group_id, newsgroup in enumerate(newsgroup_list):
      dir_path = parrent_path + '/' + newsgroup + '/'

      files = [(filename, dir_path + filename)
               for filename in listdir(dir_path)
               if isfile(dir_path + filename)]
      files.sort()
      label = group_id
      print('Processing: {}-{}'.format(group_id, newsgroup))

      for filename, filepath in files:
        with open(filepath, encoding="utf8", errors='ignore') as f:
          text = f.read().lower()
          words = re.split('\W+', text)
          if word_count is not None:
            for word in words:
              word_count[word] += 1
          content = ' '.join(words)
          assert len(content.splitlines()) == 1
          data.append(str(label) + '<fff>'
                      + filename + '<fff>' + content)
    return data

  word_count = defaultdict(int)

  path = '../datasets/20news-bydate/'
  parts = [path + dir_name + '/' for dir_name in listdir(path)
           if not isfile(path + dir_name)]

  train_path, test_path = (parts[0], parts[1]) \
                           if 'train' in parts[0] else (parts[1], parts[0])

  newsgroup_list = [newsgroup for newsgroup in listdir(train_path) if not newsgroup.startswith('.')]
  newsgroup_list.sort()

  train_data = collect_data_from(
    parrent_path=train_path,
    newsgroup_list=newsgroup_list,
    word_count=word_count
  )

  vocab = [word for word, freq in
           zip(word_count.keys(), word_count.values()) if freq > 10]
  vocab.sort()
  with open('../datasets/w2v/vocab-raw.txt', 'w') as f:
    f.write('\n'.join(vocab))

  test_data = collect_data_from(
    parrent_path=test_path,
    newsgroup_list=newsgroup_list
  )

  with open('../datasets/w2v/20news-train-raw.txt', 'w') as f:
    f.write('\n'.join(train_data))

  with open('../datasets/w2v/20news-test-raw.txt', 'w') as f:
    f.write('\n'.join(test_data))

def encode_data(data_path, vocab_path):
  with open(vocab_path) as f:
    vocab = dict([(word, word_ID + 2)
                  for word_ID, word in enumerate(f.read().splitlines())])
  with open(data_path) as f:
    documents = [(line.split('<fff>')[0], line.split('<fff>')[1], line.split('<fff>')[2])
                 for line in f.read().splitlines()]
  encoded_data = []
  for document in documents:
    label, doc_id, text = document
    words = text.split()[:MAX_DOC_LENGTH]
    sentence_length = len(words)
    encoded_text = []
    for word in words:
      if word in vocab:
        encoded_text.append(str(vocab[word]))
      else:
        encoded_text.append(str(unknown_ID))

    if len(words) < MAX_DOC_LENGTH:
      num_paddding = MAX_DOC_LENGTH - len(words)
      for _ in range(num_paddding):
        encoded_text.append(str(padding_ID))

    encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' +
                        str(sentence_length) + '<fff>' + ' '.join(encoded_text))

  dir_name = '/'.join(data_path.split('/')[:-1])
  file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
  with open(dir_name + '/' + file_name, 'w') as f:
    f.write('\n'.join(encoded_data))

class DataReader:
  def __init__(self, data_path, batch_size):
    self._batch_size = batch_size
    with open(data_path) as f:
      d_lines = f.read().splitlines()

    self._data = []
    self._labels = []
    self._sentence_length = []
    self._final_token = []
    for data_id, line in enumerate(d_lines):
      vector = []
      features = line.split('<fff>')
      label, doc_id, sentence_length = int(features[0]), int(features[1]), int(features[2])
      encoded_texts = features[3].split()
      for encoded_text in encoded_texts:
        vector.append(int(encoded_text))
      self._data.append(vector)
      self._labels.append(label)
      self._sentence_length.append(sentence_length)
      self._final_token.append(vector[-1])

    self._data = np.array(self._data)
    self._labels = np.array(self._labels)
    self._sentence_length = np.array(self._sentence_length)
    self._final_token = np.array(self._final_token)

    self._num_epoch = 0
    self._batch_id = 0

  def next_batch(self):
    start = self._batch_id * self._batch_size
    end = start + self._batch_size
    self._batch_id += 1

    if end + self._batch_size > len(self._data):
      # end = len(self._data)
      self._num_epoch += 1
      self._batch_id = 0
      indices = list(range(len(self._data)))
      np.random.seed(2018)
      np.random.shuffle(indices)
      self._data, self._labels, self._sentence_length, self._final_token = self._data[indices], self._labels[indices], self._sentence_length[indices], self._final_token[indices]

    # print(start, ' ', end)
    return self._data[start:end], self._labels[start:end], self._sentence_length[start:end], self._final_token[start:end]


class RNN:
  def __init__(self,
               vocab_size,
               embedding_size,
               lstm_size,
               # pretrained_w2v_path,
               batch_size):
    self._vocab_size = vocab_size
    self._embedding_size = embedding_size
    self._lstm_size = lstm_size
    self._batch_size = batch_size

    self._data = tf.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
    self._labels = tf.placeholder(tf.int32, shape=[batch_size, ])
    self._sentence_lengths = tf.placeholder(tf.int32, shape=[batch_size, ])
    self._final_tokens = tf.placeholder(tf.int32,
                                        shape=[batch_size, ])

  def embedding_layer(self, indices):
    pretrained_vectors = []
    pretrained_vectors.append(np.zeros(self._embedding_size))
    np.random.seed(2018)
    for _ in range(self._vocab_size + 1):
      pretrained_vectors.append(np.random.normal(loc=0, scale=1, size=self._embedding_size))

    pretrained_vectors = np.array(pretrained_vectors)

    self._embedding_matrix = tf.get_variable(
      name='embedding',
      shape=(self._vocab_size + 2, self._embedding_size),
      initializer=tf.constant_initializer(pretrained_vectors)
    )

    return tf.nn.embedding_lookup(self._embedding_matrix, indices)

  def LSTM_layer(self, embeddings):
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
    zero_state = tf.zeros(shape=(self._batch_size, self._lstm_size))
    initial_state = tf.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)

    lstm_inputs = tf.unstack(
      tf.transpose(embeddings, perm=[1, 0, 2])
    )

    lstm_ouputs, last_state = tf.nn.static_rnn(
      cell=lstm_cell,
      inputs=lstm_inputs,
      initial_state=initial_state,
      sequence_length=self._sentence_lengths
    )

    lstm_ouputs = tf.unstack(
      tf.transpose(lstm_ouputs, perm=[1, 0, 2])
    )

    lstm_ouputs = tf.concat(
      lstm_ouputs,
      axis=0
    )

    mask = tf.sequence_mask(
      lengths=self._sentence_lengths,
      maxlen=MAX_DOC_LENGTH,
      dtype=tf.float32
    )
    mask = tf.concat(tf.unstack(mask, axis=0), axis=0)
    mask = tf.expand_dims(mask, -1)

    lstm_ouputs = mask * lstm_ouputs
    lstm_ouputs_split = tf.split(lstm_ouputs, num_or_size_splits=self._batch_size)
    lstm_ouputs_sum = tf.reduce_sum(lstm_ouputs_split, axis=1)
    lstm_ouputs_average = lstm_ouputs_sum / tf.expand_dims(
      tf.cast(self._sentence_lengths, tf.float32),
      -1
    )

    return lstm_ouputs_average

  def build_graph(self):
    embeddings = self.embedding_layer(self._data)
    lstm_ouputs = self.LSTM_layer(embeddings)

    weights = tf.get_variable(
      name='final_layer_weights',
      shape=(self._lstm_size, NUM_CLASSES),
      initializer=tf.random_normal_initializer(seed=2018)
    )

    biases = tf.get_variable(
      name='final_layer_biases',
      shape=(NUM_CLASSES),
      initializer=tf.random_normal_initializer(seed=2018)
    )

    logits = tf.matmul(lstm_ouputs, weights) + biases

    labels_one_hot = tf.one_hot(
      indices=self._labels,
      depth=NUM_CLASSES,
      dtype=tf.float32
    )

    loss = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits
    )
    loss = tf.reduce_mean(loss)

    probs = tf.nn.softmax(logits)
    predicted_labels = tf.argmax(probs, axis=1)
    predicted_labels = tf.squeeze(predicted_labels)

    return predicted_labels, loss

  def trainer(self, loss, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op

def train_and_evaluate_RNN():
  with open('../datasets/w2v/vocab-raw.txt') as f:
    vocab_size = len(f.read().splitlines())

  tf.set_random_seed(2018)
  rnn = RNN(
    vocab_size=vocab_size,
    embedding_size=300,
    lstm_size=50,
    batch_size=50
  )
  predicted_labels, loss = rnn.build_graph()
  train_op = rnn.trainer(loss=loss, learning_rate=0.01)

  with tf.Session() as sess:
    train_data_reader = DataReader(
      data_path='../datasets/w2v/20news-train-encoded.txt',
      batch_size=50
    )

    test_data_reader = DataReader(
      data_path='../datasets/w2v/20news-test-encoded.txt',
      batch_size=50
    )

    step = 0
    MAX_STEP = 100 ** 2

    sess.run(tf.global_variables_initializer())
    while (step < MAX_STEP):
      next_train_batch = train_data_reader.next_batch()
      train_data, train_labels, train_sentence_lengths, train_final_tokens = next_train_batch
      plabels_eval, loss_eval, _ = sess.run(
        [predicted_labels, loss, train_op],
        feed_dict={
          rnn._data: train_data,
          rnn._labels: train_labels,
          rnn._sentence_lengths: train_sentence_lengths,
          rnn._final_tokens: train_final_tokens
        }
      )
      step += 1
      if step % 20 == 0:
        print('loss: ', loss_eval)

      if train_data_reader._batch_id == 0:
        num_true_preds = 0
        while True:
          next_test_batch = test_data_reader.next_batch()
          test_data, test_labels, test_sentence_lengths, test_final_tokens = next_test_batch

          test_plabels_eval = sess.run(
            predicted_labels,
            feed_dict={
              rnn._data: test_data,
              rnn._labels: test_labels,
              rnn._sentence_lengths: test_sentence_lengths,
              rnn._final_tokens: test_final_tokens
            }
          )
          matches = np.equal(test_plabels_eval, test_labels)
          num_true_preds += np.sum(matches.astype(float))

          if test_data_reader._batch_id == 0:
            break
        print('Epoch: ', train_data_reader._num_epoch)
        print('Accuracy on test data: ', num_true_preds * 100 / len(test_data_reader._data))


if __name__ == '__main__':
  # Xử lý dữ liệu
  # gen_data_and_vocab()
  # encode_data('../datasets/w2v/20news-train-raw.txt', '../datasets/w2v/vocab-raw.txt')
  # encode_data('../datasets/w2v/20news-test-raw.txt', '../datasets/w2v/vocab-raw.txt')

  train_and_evaluate_RNN()