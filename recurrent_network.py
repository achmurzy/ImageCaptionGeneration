'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

Modifying this script to include some MS-COCO data and start building a network from it.
'''

from __future__ import print_function

import densecap_processing as dp
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import code

# To make input, we need to create a tensor
# Our tensor will include:
# -Vector for each phrase in an image
# -Higher dimensional representation of number of words in each sequence
# -Vector representation of each word

class NetworkInput(object):

    @property
    def phrases(self):
        return self.inputs[0]

    @property
    def captions(self):
        return self.inputs[1]

    def __init__(self, phraseDim, wordDim, inputs, batchSize, epochSize, trainingIterations):
        self.phrase_dimension = phraseDim
        self.word_dimension = wordDim
        self.inputs = inputs
        self.batch_size = batchSize
        self.epoch_size = epochSize
        self.training_iterations = trainingIterations

class NetworkParameters(object):
    def __init__(self, layerSize, numLayers, learningRate):
        self.layer_size = layerSize
        self.num_layers = numLayers
        self.learning_rate = learningRate
        self.data_type = tf.float32

class LSTMNet(object):

    @property
    def inputs(self):
        return self._input

    @property
    def parameters(self):
        return self._parameters

    @property
    def model(self):
        return self._model

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def _x(self):
        return self.placeholder_x

    @property
    def _y(self):
        return self.placeholder_y

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def train_op(self):
        return self._train

    @property
    def decoder(self):
        return self._decoder

    @property
    def encoder(self):
        return self._encoder

    def __init__(self, inputs, params, codex):
        #Add network parameters objects
        self._input = inputs
        self._parameters = params
        self._decoder = codex[0]
        self._encoder = codex[1]

        #Build the LSTM network

        # tf Graph input - placeholders must be fed training data on execution
        # 'None' as a dimension allows that dimension to be any length
        self.placeholder_x = tf.placeholder(
            "float", [inputs.batch_size, inputs.phrase_dimension, inputs.word_dimension])
        self.placeholder_y = tf.placeholder("float", [None, inputs.word_dimension])
        x = tf.reshape(self._x, [-1, inputs.word_dimension])
        x = tf.split(0, inputs.batch_size, x)
        print (x)
    
        # Define an lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(
            params.layer_size, forget_bias=1.0, state_is_tuple=True)
        layer_cell = rnn_cell.MultiRNNCell([lstm_cell] * params.num_layers, state_is_tuple=True)
        
        # Save a snapshot of the initial state for generating sequences later
        self._initial_state = layer_cell.zero_state(inputs.phrase_dimension, params.data_type)
        print (self._initial_state)
        #code.interact(local=dict(globals(), **locals()))
        # Get lstm cell output - Use input split placeholder (sequence) and cell architecture
        outputs, state = rnn.rnn(
            layer_cell, x, initial_state = self._initial_state, dtype=params.data_type)
    
        #Not sure about this one
        self._final_state = state
    
        # Define weights - will generalize weight/bias function (softmax, etc.) later
        weights = {
            'out': tf.Variable(tf.random_normal([params.layer_size, inputs.word_dimension]))}
        biases = {'out': tf.Variable(tf.random_normal([inputs.word_dimension]))}

        #outputs [-1] represents the final cell in the LSTM block
        #'logits' - discrete logistic regression
        self._model =  tf.matmul(outputs[-1], weights['out']) + biases['out']

        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._model, self._y))
        self._optimizer = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate).minimize(self._cost)

    def run_epoch(self):
        """Runs the model on the given data."""
        init = tf.initialize_all_variables()
        with tf.Session() as session:
            session.run(init)
            costs = 0.0
            iters = 0
            state = session.run(self.initial_state)
            fetches = {"cost": self.cost, "final_state": self.final_state, 
                           "eval_op":self.optimizer}
            for step in range(self.inputs.training_iterations):
                train_dict = {self._x: self.inputs.phrases, 
                                      self._y: self.inputs.captions}
                #code.interact(local=dict(globals(), **locals()))
                vals = session.run(fetches, train_dict)
                cost = vals["cost"]
                state = vals["final_state"]
                probs = session.run(self.model, train_dict)
                print(self.sample(probs))
                costs += cost

        return np.exp(costs)

    def sample(self, probabilities):
        gen = []
        cap = ""
        for prob in probabilities:
            ind = np.argmax(prob)
            gen.append(ind)
            cap += self.decoder[ind]
        return cap

def MakePlaceholderTensors(phraseCount, phraseLength, wordCount):
    # tf Graph input - placeholders must be fed training data on execution
    # 'None' as a dimension allows that dimension to be any length    Can we use an arbitraty
    # x = tf.placeholder("float", [None, phraseLength, wordCount])    number of phrases this way?
    x = tf.placeholder("float", [phraseCount, phraseLength, wordCount])

    # Are we forced to define fixed-length captions to provide training labels?
    # No. We generate a caption each step and use sequence-length independent
    # machien translation metrics to define a cost function
    # y = tf.placeholder("float", [None, wordCount])  
    y = tf.placeholder("float", [None, wordCount])  
    
    return x, y

#creates weight vector and bias vector applied to all cells in a layer
def MakeLayerParameters(layerCount, wordCount):
    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([layerCount, wordCount]))
    }

    biases = {
    'out': tf.Variable(tf.random_normal([wordCount]))
    }
    return weights, biases


def RNN(x, phraseCount, phraseLength, wordCount, n_hidden, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (phraseLength, phraseCount, wordCount)
    # Required shape: 'phraseCount' tensors list of shape (phraseLength, wordCount)

    print("Input tensor: ", x)
    # Permuting batch_size and phraseLength
    # x = tf.transpose(x, [1, 0, 2])

    # print("Transpose tensor: ", x)
    # Reshaping to (phraseCount*phraseLength, wordCount)
    # -1 infers dimension based on specified dimension

    x = tf.reshape(x, [-1, wordCount])
    print("Reshaped tensor: ", x)
    
    # Split to get a list of 'phraseCount' tensors of shape (phraseLength, wordCount)
    print("Split dimension: ", tf.shape(x)) 
    x = tf.split(0, phraseCount, x)
    print("Split tensor: ", x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    
    print("LSTM Output: ", outputs)
    print("LSTM State Tuple 'c': ", states.c) 
    print("LSTM State Tuple 'h': ", states.h)

    # Linear activation, using rnn inner loop last output
    #This is the model
    code.interact(local=dict(globals(), **locals()))
    #code.interact(local = locals())
    #outputs [-1] represents the final cell in the LSTM block
    #building a model wrt this cell gives us our output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Define loss and optimizer to perform gradient descent
# Takes a network, an output structure (tensor y) and a learning rate 
def LossFunction(pred, y, learning_rate):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return cost, optimizer

# Evaluate model
# Used to feed back model accuracy during and after training
def AccuracyFunction(pred, y):
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def RunModel(pred, x, y, phrases, captions, optimizer, cost, accuracy, training_iters, display_step, invertDict):
    #input_phrases = tf.Variable(x, trainable=False, collections=[])
    #input_captions = tf.Variable(y, trainable=False, collections=[])

    #input_phrases = tf.constant(phrases)
    #input_captions = tf.constant(captions)
    
    #print (input_phrases)
    #print (input_captions)
    
    #print (phrases)
    #print (captions)

    #phrase, caption = tf.train.slice_input_producer(
    #    [input_phrases, input_captions], num_epochs = training_iters)

    #print ("Phrase shape: %s") % (tf.shape(phrases))

    # Initializing the variables
    init = tf.initialize_all_variables()
    
    # Launch the graph
    with tf.Session() as sess:
        runResult = sess.run(init)
        step = 0
        # Keep training until reach max iterations
        # Each step is one image, associated with five phrases and a caption
        while step < training_iters:
            # Run optimization op (backprop)
            print("Optimizing...")
            #print("Phrases: ", phrases)
            trainDict = {x: phrases, y: captions}
            print (trainDict)
            output = sess.run(optimizer, feed_dict=trainDict)
            probs = sess.run(pred, feed_dict=trainDict)
            print (probs)
            print (generate_caption(probs, invertDict))
            #code.interact(local=dict(globals(), **locals()))
            '''if step % display_step == 0:
                # caption = generate_caption()
                # Calculate batch accuracy
                #acc = sess.run(accuracy, feed_dict={x: caption, y: captions})
                acc = sess.run(accuracy, feed_dict={x: phrases, y: captions})
                # Calculate batch loss
                #loss = sess.run(cost, feed_dict={x: caption, y: captions})
                loss = sess.run(cost, feed_dict={x: phrases, y: captions})
                print("Iter " + str(step) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))'''
            step += 1
        print("Optimization Finished!")
    return sess
    #closure = tf.Session.close(sess)
    #print (runResult)
    #print (closure)
    

def generate_caption(probabilities, invert):
    gen = []
    cap = ""
    for prob in probabilities:
        ind = np.argmax(prob)
        gen.append(ind)
        cap += invert[ind]
    return cap

'''def fill_feed_dict(data_set, phrases_pl, captions_pl):
  Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of phrases and captions
    phrases_pl: The phrases placeholder.
    captions_pl: The captions placeholder.
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  
  # Create the feed_dict for the placeholders
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
return feed_dict'''
