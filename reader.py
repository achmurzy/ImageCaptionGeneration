# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import code
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_phrase_data, raw_caption_data, batch_size, phrase_count, phrase_length, lexDim, name=None):
  """Iterate on the raw PTB data.

  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    phrase_count: int, the phrase count.
    phrase_length: int, the number of unrolls.
    name: the name of this operation (optional).

  Returns:
    A pair of Tensors, each shaped [phrase_count, phrase_length]. The second element
    of the tuple is the same data time-shifted to the right by one.

  Raises:
    tf.errors.InvalidArgumentError: if phrase_count or phrase_length are too high.
  """
  with tf.name_scope("PTBProducer"):
    raw_phrase_data = tf.convert_to_tensor(raw_phrase_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_phrase_data)

    batch_len = (batch_size * phrase_length * lexDim)
    epoch_size = (data_len) // (batch_len)

    #input_imgs = data_len // (phrase_length * lexDim)
    #data = tf.reshape(raw_phrase_data[0 : data_len],
    #                 [input_imgs, phrase_length, lexDim])

    #batch_len = data_len // (batch_size * phrase_length * lexDim)
    data = tf.reshape(raw_phrase_data[0 : epoch_size * batch_len],
                      [epoch_size * batch_size, phrase_length, lexDim])
    
     
    ###epoch_size = batch_len // (phrase_length * lexDim)

    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease phrase_count or phrase_length")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    raw_caption_data = tf.convert_to_tensor(raw_caption_data, name="raw_captions", dtype=tf.int32)
    caption_data_len = tf.size(raw_caption_data)

    #caption_data = tf.reshape(raw_caption_data[0: caption_data_len],
    #                          [inputs_imgs, phrase_length,  lexDim])

    caption_data = tf.reshape(raw_caption_data[0: epoch_size * batch_len],
                              [epoch_size * batch_size, phrase_length,  lexDim])
               
               
    #Iteratively dequeues integers in the range of iterations of an epoch 
    epoch_queue = tf.train.range_input_producer(epoch_size, shuffle=False)
    i = epoch_queue.dequeue()
    #Accesses data by slicing with the asynchronously updated epoch index

    #x = tf.slice(data, [i, 0, 0], [batch_size, phrase_length, lexDim])
    #x = tf.squeeze(x, [0])
    x = tf.slice(data, [i*batch_size, 0, 0], [batch_size, phrase_length, lexDim])

    #y = tf.slice(caption_data, [i, 0, 0], [batch_size, phrase_length, lexDim])
    #y = tf.squeeze(y, [0])
    y = tf.slice(caption_data, [i*batch_size, 0, 0], [batch_size, phrase_length, lexDim])

    print ("At reader")
    code.interact(local=dict(globals(), **locals()))
    return x, y, epoch_size

def ptb_id_producer(raw_phrase_data, raw_caption_data, batch_size, phrase_count, phrase_length, name=None):
  
  with tf.name_scope("PTB_id_Producer"):
    raw_phrase_data = tf.convert_to_tensor(raw_phrase_data, name="raw_data", dtype=tf.int32)
    data_len = tf.size(raw_phrase_data)

    batch_len = (batch_size * phrase_count * phrase_length)
    epoch_size = (data_len) // (batch_len)
    data = tf.reshape(raw_phrase_data[0 : epoch_size * batch_len],
                      [batch_size * epoch_size, phrase_count, phrase_length])
    
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease phrase_count or phrase_length")
    with tf.control_dependencies([assertion]):
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    raw_caption_data = tf.convert_to_tensor(raw_caption_data, name="raw_captions", dtype=tf.int32)
    caption_data_len = tf.size(raw_caption_data)

    caption_data = tf.reshape(raw_caption_data[0: epoch_size * batch_len],
                              [batch_size * epoch_size, phrase_count, phrase_length])
               
               
    #Iteratively dequeues integers in the range of iterations of an epoch 
    epoch_queue = tf.train.range_input_producer(epoch_size, shuffle=False)
    i = epoch_queue.dequeue()

    #Accesses data by slicing with the asynchronously updated epoch index
    x = tf.slice(data, [i*batch_size, 0], [batch_size, phrase_count, phrase_length])
    y = tf.slice(caption_data, [i*batch_size, 0], [batch_size, phrase_count, phrase_length])

    print ("At reader")
    code.interact(local=dict(globals(), **locals()))
    return x, y, epoch_size
