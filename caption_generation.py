import densecap_processing as dp
import recurrent_network as rn
import sys
import code
import re

#This script gets five images from MS-COCO, and transforms
#densecap phrases and training phrases into a vectorized form
#we can input to our RNN

##########MS-COCO TRAINING CAPTION EXTRACTION##############
training_iters = 1

#get three img IDs from MS-COCO
imgIDs = dp.get_coco_imgs(training_iters)

#get caption sets for each image
capDict = dp.coco_to_captions(imgIDs)

#get one training caption per image
captions = dp.get_coco_captions(capDict)

#Build a lexicon and encoder/decoder dictionaries
lex = dp.get_coco_lexicon(capDict)
wordDict = {}
invertDict = {}
dp.build_lookup_lexicon(lex, wordDict, invertDict)

###########DENSECAP PHRASE EXTRACTION######################
#Use if images need re-processing by densecap
#Run as python caption_generation.py 1
'''if(len(sys.argv) > 1):
    processImages = int(sys.argv[1])
    if(processImages):
        dp.coco_to_densecap(imgIDs)'''

#Get densecap results
results = dp.json_to_dict("results/results.json")
image_props = dp.dict_to_imgs(results)
    
###########NETWORK CONSTRUCTION AND EXECUTION################

# Network Parameters 
n_hidden = 64 # hidden layer num of features (# of 'neurons')
n_layers = 5 # number of stacked layers - should equal number of phrases (so batch size?)
learning_rate = 0.001

# Input Parameters
batch_size = 5 # of densecap phrases to use in tensor input per epoch
phraseLength = 5 # of words per phrase. This will become a function of phrase inputs
LEX_DIM = (len(wordDict))
epoch_size = 1
display_step = 1

captions = dp.extract_caption_vectors(phraseLength, invertDict, captions)
phrases = dp.extract_phrase_vectors(
    phraseLength, batch_size, training_iters, image_props, invertDict)
#code.interact(local=dict(globals(), **locals()))
inputs = rn.NetworkInput(batch_size, phraseLength, LEX_DIM, [phrases, captions], epoch_size, training_iters)
params = rn.NetworkParameters(n_hidden, n_layers, learning_rate) 
ann = rn.LSTMNet(inputs, params, [wordDict, invertDict])
ann.run_epoch()
