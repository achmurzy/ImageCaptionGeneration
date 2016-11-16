'''
gotta be bad to get good
'''

from __future__ import print_function

import densecap_processing as dp
import reader

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.contrib.slim.python.slim import evaluation

import numpy as np
#from nltk import bleu_score
import code
import matplotlib.pyplot as plt


# To make input, we need to create a tensor
# Our tensor will include:
# -Vector for each phrase in an image
# -Higher dimensional representation of number of words in each sequence
# -Vector representation of each word

class NetworkInput(object):

    @property
    def phrases(self):
        return self.phrase_batch
    @phrases.setter
    def phrases(self, value):
        self.phrase_batch = value

    @property
    def captions(self):
        return self.caption_batch
    @captions.setter
    def captions(self, value):
        self.caption_batch = value

    def __init__(self, batchSize, phraseCount, phraseDim, wordDim, phraseBatch, captionBatch, numEpochs, inputImgLength):
        self.phrase_count = phraseCount
        self.phrase_dimension = phraseDim
        self.word_dimension = wordDim
        #self.inputs = [np.asarray(inputs[0], dtype=np.float32), 
        #               np.asarray(inputs[1], dtype=np.float32)] 
        self.batch_size = batchSize
        self.num_epochs = numEpochs
        self.data_size = inputImgLength
        self.phrase_batch = phraseBatch
        self.caption_batch = captionBatch
        
class NetworkParameters(object):
    def __init__(self, layerSize, numLayers, learningRate, initScale):
        self.layer_size = layerSize
        self.num_layers = numLayers
        self.learning_rate = learningRate
        self.data_type = tf.float32
        self.init_scale = initScale

class NetworkResults(object):
    def __init__(self, displayStep):
        self.costHistory = {}
        self.display_step = displayStep

    def record_point(self, key, val):
        self.costHistory[key] = val
        
    def plot_results(self):
        plt.plot(self.costHistory.keys(), self.costHistory.values())
        plt.xticks(np.arange(min(self.costHistory.keys()), 
                             max(self.costHistory.keys()), self.display_step))
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.tick_params(axis='both', which='minor', labelsize=8)
        plt.show()
        

class LSTMNet(object):

    @property
    def inputs(self):
        return self._input

    @property
    def parameters(self):
        return self._parameters

    @property
    def results(self):
        return self._results

    @property
    def log_path(self):
        return "results/log"

    @property
    def global_step(self):
        return self.globalStep

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
    def training_iterations(self):
        return self.inputs.data_size / self.inputs.batch_size
    #def training_iterations(self, session):
    #    return self.inputs.epoch_size.eval(session=session)

    def __init__(self, inputs, params, results, codex):
        with tf.variable_scope("Model", reuse=None):

            #Add network parameters objects
            self._input = inputs
            self._parameters = params
            self._results = results

            self._decoder = codex[0]
            self._encoder = codex[1]

            self.epochs = 0
            self.epoch_iteration = 0

            #####################Build the LSTM network#############################

            # tf Graph input - placeholders must be fed training data on execution
            # 'None' as a dimension allows that dimension to be any length

            #These placeholders need to be integer-typed to use embeddings
            #embeddings allow us to encode data for the hidden layer

            #self.placeholder_x = tf.placeholder(tf.int32, 
            #[inputs.phrase_count, inputs.phrase_dimension, inputs.word_dimension])

            with tf.variable_scope("Inputs"):
                self.placeholder_x = self.inputs.phrases
                #code.interact(local=dict(globals(), **locals()))
                #self.placeholder_x = tf.placeholder(params.data_type, 
                #[inputs.batch_size, inputs.phrase_count, inputs.phrase_dimension, inputs.word_dimension])

                self.placeholder_y = self.inputs.captions
                #self.placeholder_y = tf.placeholder(params.data_type, 
                #                    [inputs.phrase_dimension, inputs.word_dimension])
                #self.placeholder_y = tf.placeholder(params.data_type, 
                #[inputs.batch_size, inputs.phrase_dimension, inputs.word_dimension])

                #x = tf.reshape(self._x, [inputs.phrase_count, -1]) #Define sequence length of
                #x = tf.split(0, inputs.phrase_count, self._x)         # RNN input - "unroll" the network
                #x = [tf.squeeze(_x, [0]) for _x in x]

            with tf.variable_scope("Input_Layer"):
                with tf.device("/cpu:0"):
                    #embedding = tf.get_variable(
                    #    "embedding", [inputs.word_dimension, params.layer_size], dtype=params.data_type)
                    embedding = tf.get_variable("embedding", 
                            [inputs.phrase_dimension, params.layer_size], dtype=params.data_type)
                    phraseEmbedding = tf.nn.embedding_lookup(embedding, self._x)
                #phraseEmbedding = [tf.squeeze(input_step, [0])
                #    for input_step in tf.split(0, inputs.phrase_count, phraseEmbedding)]
                code.interact(local=dict(globals(), **locals()))
                phraseEmbedding = [tf.squeeze(input_step, [0])
                    for input_step in tf.split(0, inputs.phrase_dimension, phraseEmbedding)]
                code.interact(local=dict(globals(), **locals()))

            # Define an lstm cell with tensorflow
            lstm_cell = rnn_cell.BasicLSTMCell(
                params.layer_size, forget_bias=1.0, state_is_tuple=True)
            layer_cell = rnn_cell.MultiRNNCell(
                [lstm_cell] * params.num_layers, state_is_tuple=True)

            with tf.name_scope("Initial_State"):
                # Save a snapshot of the initial state for generating sequences later
                #self._initial_state = layer_cell.zero_state(inputs.batch_size, params.data_type)
                self._initial_state = layer_cell.zero_state(
                    inputs.word_dimension, params.data_type)
            
            #code.interact(local=dict(globals(), **locals()))
            outputs, state = rnn.rnn(
                layer_cell, phraseEmbedding, 
                initial_state = self._initial_state, dtype=params.data_type)
            code.interact(local=dict(globals(), **locals()))

            with tf.name_scope("Final_State"):
                #Used as recurrent input to LSTM layers during sequence generation
                #Represents (c, h) values for params.num_layer of stacked LSTM cells
                #(Given as 'unrolled' representation) - Also used as input to model by determining
                # value of outputs[-1] - therefore directly used to compute probabilities
                self._final_state = state

            # Define weights according to dimensionality of hidden layers
            # Randomly initializing weights and biases ensures feature differentiation
            #weights = {
            #    'out': tf.Variable(tf.random_normal([params.layer_size, inputs.word_dimension]))}
            #biases = {'out': tf.Variable(tf.random_normal([inputs.word_dimension]))}

            with tf.variable_scope("Output_Layer"):

                with tf.variable_scope("Squash"):
                    weights = {
                        'out': tf.Variable(tf.random_normal([params.layer_size, 1]))}
                    biases = {'out': tf.Variable(tf.random_normal([inputs.word_dimension]))}

                    #outputs [-1] represents the final cell in the LSTM block, given as batch_size
                    #of tensors for handling output
                    #Output in LSTM is a function of the cell state (c) and the hidden state (h)
                    #See LSTMStateTuple output of rnn_cell.BasicLSTMCell (state)

                    #This represents the model we apply to the LSTM cell layer
                    #This reconciles the dimensionality of hidden features (layer_size) and LSTM states
                    #with dimensionality of our sequence (phrase_dim, word_dim)
                    #Returns sequence predictions - These values are used for classification
                    self._model =  tf.add(tf.squeeze(
                        tf.matmul(outputs[-1], weights['out'])), biases['out'])
                    #self._model =  tf.mul(outputs[-1], weights['out']) + biases['out']

                    #self.probabilities is the final layer of the network
                    #squash all predictions into range 0->1 for sane inference 
                    self._probs = tf.nn.softmax(self._model)

                with tf.variable_scope("Backpropagation"):
                    self._cost = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(self._model, self._y))
                    #logits = tf.split(0, inputs.batch_size, tf.reshape(
                    #                self._x, [inputs.batch_size, -1]))
                    #targets = [self._y] * inputs.batch_size
                    #weights = [tf.ones(self.inputs.batch_size * inputs.word_dimension, 
                    #     dtype=params.data_type)] * inputs.batch_size
                    #self._cost = tf.reduce_mean(
                    #tf.nn.seq2seq.sequence_loss(logits,targets,weights))

                    #I don't know what this does. Some variant of backpropagation
                    self.globalStep = tf.Variable(0, name='global_step', trainable=False)
                    self._optimizer = tf.train.AdamOptimizer(
                        learning_rate=params.learning_rate).minimize(
                            self._cost, global_step=self.globalStep)
                    #code.interact(local=dict(globals(), **locals()))

    def train_network(self):
        initializer = tf.initialize_all_variables()
        #initializer=tf.random_uniform_initializer(-self.parameters.init_scale, 
        ###                                          self.parameters.init_scale)
        #Supervisor does nice things, like start queues
        saves = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        sv = tf.train.Supervisor(logdir=self.log_path, saver=saves, 
                                 global_step=self.global_step, save_model_secs=1)      
        with sv.managed_session() as session:
            session.run(initializer)
            while self.epochs <= self.inputs.num_epochs:
                self.run_epoch(session)
        #with tf.Session() as session:
        #    print(self.sample(session))  # -- get caption
        self.results.plot_results()

    def run_epoch(self, session):
        """Runs the model on the given data."""
        costs = 0.0
        state = session.run(self.initial_state)
        fetches = {"cost": self.cost, "final_state": self.final_state, 
                       "eval_op":self.optimizer}
        print ("Epoch: ", self.epochs)
        #code.interact(local=dict(globals(), **locals()))
        for step in range(self.training_iterations):
            print (step)
            #phrases, captions = self.next_batch()
            #train_dict = {self._x: phrases, self._y: captions}                
            train_dict = {}
            for i, (c, h) in enumerate(self.initial_state): #fails, placeholders are empty 
                train_dict[c] = state[i].c
                train_dict[h] = state[i].h
            #code.interact(local=dict(globals(), **locals()))  
            vals = session.run(fetches, train_dict)
            #Equivalent to:                         Against input x, y (phrases, captions)
            '''session.run(self.final_state, train_dict) #Compute probability distribution
            code.interact(local=dict(globals(), **locals()))
            session.run(self.cost, train_dict)        #Calculate loss
            code.interact(local=dict(globals(), **locals()))
            session.run(self.optimizer, train_dict)   #Update weights by backpropagation'''
            
            cost = vals["cost"]
            state = vals["final_state"]
            costs += cost
        print ("Cost: ", costs)
        if self.epochs % self.results.display_step == 0:
            self.results.record_point(self.epochs, costs)
        self.epochs += 1
        print(self.globalStep.eval(session))
        return np.exp(costs)

    def sample(self, session, seed = '\''):
        state = session.run(self.initial_state)  #See constructor - tensor of 0's
        #placeholder_x = tf.placeholder(tf.int32, 
        #                                   [self.inputs.phrase_dimension, self.inputs.word_dimension])
        ret = seed
        char = ret
        code.interact(local=dict(globals(), **locals()))
        x = np.zeros((self.inputs.phrase_dimension, self.inputs.word_dimension))
        #x = np.zeros((self.inputs.phrase_count, 
        #              self.inputs.phrase_dimension, self.inputs.word_dimension))
        print ("loopin")
        x[0, 0] = self.encoder[char]
        feed = {self._x: x, self.initial_state:state}
        [state] = session.run([self.final_state], feed)
        code.interact(local=dict(globals(), **locals()))
        #Seed state is trained network on <START> = '\''
        
        num = self.inputs.phrase_dimension #For now, fixed length captions
        for n in range(num):              #Ideally this loop is 'until generate <STOP>'
            x = np.zeros((self.inputs.phrase_dimension, self.inputs.word_dimension))
            #x = np.zeros((self.inputs.phrase_count, 
            #              self.inputs.phrase_dimension, self.inputs.word_dimension))

            x[0, 0] = self.encoder[char]
            feed = {self._x: x, self.initial_state:state}
            [probs, state] = session.run([self.probabilities, self.final_state], feed)
            #p = probs[0]                #We only sample the next word in the sequence   
            p = probs
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
    
        if(self.epoch_iteration >= self.inputs.data_size):
            self.epoch_iteration = 0
            #Google has it shuffling inputs here          Could affect prediction/classification?
            #to avoid fitting the order of images in the training set?
            #code.interact(local=dict(globals(), **locals()))
            shuffleIndices = np.arange(self.data_size)
            np.random.shuffle(shuffleIndices)
            self.inputs.phrases = self.inputs.phrases[shuffleIndices]
            self.inputs.captions = self.inputs.captions[shuffleIndices]

        return phrase_batch[0], caption_batch[0]
        
        
