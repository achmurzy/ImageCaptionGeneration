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
from nltk import bleu_score
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
    @phrases.setter
    def phrases(self, value):
        self.inputs[0] = value

    @property
    def captions(self):
        return self.inputs[1]
    @captions.setter
    def captions(self, value):
        self.inputs[1] = value

    def __init__(self, batchSize, phraseCount, phraseDim, wordDim, inputs, numEpochs):
        self.phrase_count = phraseCount
        self.phrase_dimension = phraseDim
        self.word_dimension = wordDim
        self.inputs = [np.asarray(inputs[0], dtype=np.float32), 
                       np.asarray(inputs[1], dtype=np.float32)] 
        self.batch_size = batchSize
        self.num_epochs = numEpochs

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
    def probabilities(self):
        return self._probs

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

    @property
    def data_size(self):
        return len(self.inputs.phrases)

    @property 
    def training_iterations(self):
        return self.data_size / self.inputs.batch_size

    def __init__(self, inputs, params, codex):
        #Add network parameters objects
        self._input = inputs
        self._parameters = params
        self._decoder = codex[0]
        self._encoder = codex[1]

        self.epoch_iteration = 0
        

        #####################Build the LSTM network#############################

        # tf Graph input - placeholders must be fed training data on execution
        # 'None' as a dimension allows that dimension to be any length
        self.placeholder_x = tf.placeholder(params.data_type, 
        [inputs.phrase_count, inputs.phrase_dimension, inputs.word_dimension])
        #self.placeholder_x = tf.placeholder(params.data_type, 
        #[inputs.batch_size, inputs.phrase_count, inputs.phrase_dimension, inputs.word_dimension])
        
        self.placeholder_y = tf.placeholder(params.data_type, 
                            [inputs.phrase_dimension, inputs.word_dimension])
        #self.placeholder_y = tf.placeholder(params.data_type, 
        #[inputs.batch_size, inputs.phrase_dimension, inputs.word_dimension])
        
        #x = tf.reshape(self._x, [ inputs.batch_size*inputs.phrase_dimension, -1])
        #x = tf.split(0, inputs.batch_size, x) 
        x = tf.reshape(self._x, [-1, inputs.word_dimension])
        x = tf.split(0, inputs.phrase_dimension, x) 
        
        # Define an lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(
            params.layer_size, forget_bias=1.0, state_is_tuple=True)
        layer_cell = rnn_cell.MultiRNNCell([lstm_cell] * params.num_layers, state_is_tuple=True)
        
        # Save a snapshot of the initial state for generating sequences later
        self._initial_state = layer_cell.zero_state(inputs.phrase_dimension, params.data_type)
        
        outputs, state = rnn.rnn(
            layer_cell, x, initial_state = self._initial_state, dtype=params.data_type)
    
        #Used as recurrent input to LSTM layers during sequence generation
        #Represents (c, h) values for params.num_layer of stacked LSTM cells
        #(Given as 'unrolled' representation) - Also used as input to model by determining
        # value of outputs[-1] - therefore directly used to compute probabilities
        self._final_state = state
    
        # Define weights according to dimensionality of hidden layers
        # Randomly initializing weights and biases ensures feature differentiation
        weights = {
            'out': tf.Variable(tf.random_normal([params.layer_size, inputs.word_dimension]))}
        biases = {'out': tf.Variable(tf.random_normal([inputs.word_dimension]))}

        #outputs [-1] represents the final cell in the LSTM block, given as batch_size
        #of tensors for handling output
        #Output in LSTM is a function of the cell state (c) and the hidden state (h)
        #See LSTMStateTuple output of rnn_cell.BasicLSTMCell (state)
        
        #This represents the model we apply to the LSTM cell layer
        #This reconciles the dimensionality of hidden features (layer_size) and LSTM states
        #with dimensionality of our sequence (phrase_dim, word_dim)
        #Returns sequence predictions - These values are used for classification
        self._model =  tf.matmul(outputs[-1], weights['out']) + biases['out']
        
        #self.probabilities is the final layer of the network
        #squash all predictions into range 0->1 for sane inference 
        self._probs = tf.nn.softmax(self._model)
        #code.interact(local=dict(globals(), **locals()))
        #self._cost will become a custom machine translation heuristic and other things yo
        self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self._model, self._y))
        #logits = tf.split(0, inputs.batch_size, tf.reshape(
        #                self._x, [inputs.batch_size, -1]))
        #targets = [self._y] * inputs.batch_size
        #weights = [tf.ones(self.inputs.batch_size * inputs.word_dimension, 
        #     dtype=params.data_type)] * inputs.batch_size
        
        #self._cost = tf.reduce_mean(tf.nn.seq2seq.sequence_loss(logits,targets,weights))
        
        #I don't know what this does. Some variant of backpropagation
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
            
            for step in range(self.training_iterations):
                phrases, captions = self.next_batch()
                train_dict = {self._x: phrases, self._y: captions}                
                vals = session.run(fetches, train_dict)
                """Equivalent to:                         Against input x, y (phrases, captions)
                session.run(self.final_state, train_dict) Compute probability distribution
                session.run(self.cost, train_dict)        Calculate loss
                session.run(self.optimizer, train_dict)   Update weights by backpropagation
                """
                cost = vals["cost"]
                state = vals["final_state"]
                
                self.sample(session)  # -- get caption
                costs += cost

        return np.exp(costs)

    def sample(self, session, seed = '\''):
        state = session.run(self.initial_state)  #See constructor - tensor of 0's
        for char in seed[:-1]:
            #x = np.zeros((self.inputs.batch_size, self.inputs.phrase_count, 
            #              self.inputs.phrase_dimension, self.inputs.word_dimension))
            x = np.zeros((self.inputs.phrase_count, 
                          self.inputs.phrase_dimension, self.inputs.word_dimension))
            
            x[0, 0] = self.encoder[char]
            feed = {self._x: x, self.initial_state:state}
            [state] = session.run([self.final_state], feed)
            
        #Seed state is trained network on <START> = '\''
        ret = seed
        char = seed[-1]
        num = self.inputs.phrase_dimension #For now, fixed length captions
        for n in range(num):              #Ideally this loop is 'until generate <STOP>'
            #x = np.zeros((self.inputs.batch_size, self.inputs.phrase_count, 
            #              self.inputs.phrase_dimension, self.inputs.word_dimension))
            x = np.zeros((self.inputs.phrase_count, 
                          self.inputs.phrase_dimension, self.inputs.word_dimension))
            
            x[0, 0] = self.encoder[char]
            print (char)
            feed = {self._x: x, self.initial_state:state}
            [probs, state] = session.run([self.probabilities, self.final_state], feed)
            p = probs[0]                #We only sample the next word in the sequence   
            sample = np.argmax(p)          #We can write more complicated sampling functions
            pred = self.decoder[sample]
            ret += pred
            char = pred
        return ret

    #Retrieve next set of examples based on batch size - or right now, phraseCount
    def next_batch(self):
        start = self.epoch_iteration
        self.epoch_iteration += self.inputs.batch_size
        end = self.epoch_iteration
        phrase_batch = self.inputs.phrases[start:end]
        caption_batch = self.inputs.captions[start:end]
    
        if(self.epoch_iteration >= self.data_size):
            self.epoch_iteration = 0
            #Google has it shuffling inputs here          Could affect prediction/classification?
            #to avoid fitting the order of images in the training set?
            #code.interact(local=dict(globals(), **locals()))
            shuffleIndices = np.arange(self.data_size)
            np.random.shuffle(shuffleIndices)
            self.inputs.phrases = self.inputs.phrases[shuffleIndices]
            self.inputs.captions = self.inputs.captions[shuffleIndices]

        return phrase_batch[0], caption_batch[0]
        
        
