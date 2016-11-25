import densecap_processing as dp
import recurrent_network as rn
import sys
import code
import re

#Use if images need re-processing by densecap
#Run as python caption_generation.py 1
processImages = int(sys.argv[1])
train = int(sys.argv[2])

##########MS-COCO TRAINING CAPTION EXTRACTION##############
dp.set_coco_dataset(train)

inputImgCount = 50

#get three img IDs from MS-COCO
imgIDs = dp.get_coco_imgs(inputImgCount)

imgFiles = {}
invertFiles = {}
for x in imgIDs:
    if train:
        name = 'COCO_train2014_%s.jpg'%(str(x).zfill(12))
    else:
        name = 'COCO_val2014_%s.jpg'%(str(x).zfill(12))
    imgFiles[x] = name
    invertFiles[name] = x

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

if(processImages):
    dp.coco_to_densecap(imgIDs)

###########NETWORK CONSTRUCTION AND EXECUTION################

from itertools import chain
def flatten(listOfLists):
    return list(chain(*listOfLists))

# Network Parameters 
n_hidden = 64 # hidden layer num of features (# of 'neurons')
n_layers = 1 # number of stacked layers
learning_rate = 0.001 # SGD magnitude
initializationScale = 0.1 # scale of weight intializations

# Input Parameters
batch_size = 1 # of images to show per training iteration
phraseCount = 5 # of densecap phrases to use in tensor input per epoch
phraseLength = 10 # of words per phrase. This will become a function of phrase inputs
LEX_DIM = (len(wordDict))
num_epochs = 100
display_step = 2

if train:
    results = dp.json_to_dict("results/train_results.json")
else:
    results = dp.json_to_dict("results/val_results.json")
    
image_props = dp.dict_to_imgs(results)

phraseCapCorrespondence = {}
for x in image_props.keys():
    name = image_props[x]['img_name']
    if name in imgFiles.values():
        phraseCapCorrespondence[invertFiles[name]] = x

flatPhraseIDs = dp.extract_phrase_id_vectors(phraseCount, phraseLength, inputImgCount, phraseCapCorrespondence, image_props, invertDict)
flatCaptionIDs = dp.extract_caption_id_vectors(phraseLength, inputImgCount, invertDict, captions)

code.interact(local=dict(globals(), **locals()))

import reader
batchedPhrases, batchedCaptions, epochSize = reader.ptb_id_producer(
flatPhraseIDs, flatCaptionIDs, batch_size, phraseCount, phraseLength)

inputs = rn.NetworkInput(batch_size, phraseCount, phraseLength, LEX_DIM, batchedPhrases, batchedCaptions, num_epochs, epochSize)
params = rn.NetworkParameters(n_hidden, n_layers, learning_rate, initializationScale)
results = rn.NetworkResults(display_step)

ann = rn.LSTMNet(inputs, params, results, [wordDict, invertDict], train)
ann.run_network(train)
