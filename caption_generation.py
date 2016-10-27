import densecap_processing as dp
import recurrent_network as rn
import salience
import word2vec
import re
import sys
import code

#This script gets five images from MS-COCO, and transforms
#densecap phrases and training phrases into a vectorized form
#we can input to our RNN

##########MS-COCO TRAINING CAPTION EXTRACTION##############
training_iters = 5

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

############INPUT PREPROCESSING#############################

LEX_DIM = (len(wordDict) + 2)
def empty_one_hot_vector():
    return [0] * LEX_DIM    #2 - Account for START and STOP symbols

#Takes a string and uses the lexicon to convert to a one-hot
#[0, ..., 1, ..., 0] vector representation of a word
def string_to_vector(word):
    one_hot = empty_one_hot_vector()  
    one_hot[invertDict[word]] = 1
    return one_hot
 
def extract_flat_phrase_vectors():
    one_hot_list = []
    for x in range(0, training_iters):
        salient = salience.salient_phrases(
            image_props, x, lambda: salience.k_top_scores(image_props[x], batch_size))
        print "Salient phrases: ", len(salient)
        for phrase in salient:
            count = 0
            #print "Split ", len(phrase.split())
            for word in phrase.split():
                #print word
                if count >= phraseLength:
                    break
                elif word in invertDict:
                    #print "To one-hot"
                    one_hot_list.append(string_to_vector(word))
                else:
                    #print "Phrase lexicon out of bounds"
                    one_hot_list.append(empty_one_hot_vector())
                count = count + 1
            if count < phraseLength:
                one_hot_list.append(empty_one_hot_vector())
                count = count + 1
    return one_hot_list

def extract_phrase_vectors():
    one_hot_list = [[[]] * phraseLength] * batch_size
    for x in range(0, training_iters):
        salient = salience.salient_phrases(
            image_props, x, lambda: salience.k_top_scores(image_props[x], batch_size))
        phraseI = 0
        for phrase in salient:
            count = 0
            print ("Phrase ", phrase)
            for word in phrase.split():
                #print word
                if count >= phraseLength:
                    print "Break phrase"
                    break
                elif word in invertDict:
                    print "Add word"
                    one_hot_list[phraseI][count] = string_to_vector(word)
                else:
                    print "Fill void"
                    one_hot_list[phraseI][count] = empty_one_hot_vector()
                print count
                count = count + 1
            while count < phraseLength:
                one_hot_list.append(empty_one_hot_vector())
                count = count + 1
            print count
            print phraseI
            phraseI = phraseI + 1
    return one_hot_list

def extract_caption_vectors():
    one_hot_list = []
    for cap in captions:
        #print captions[cap]  #Regex parses START, STOP, preserves delimiters
        count = 0
        for word in re.split('[(\'\.\s)]', captions[cap]): 
            #print word
            if count >= phraseLength:
                print "break cap"
                break
            elif word in invertDict:
                print "To one-hot"
                one_hot_list.append(string_to_vector(word))
            else:
                #print "Caption split deviant"
                one_hot_list.append(empty_one_hot_vector())
            count = count + 1
    return one_hot_list
    
###########NETWORK CONSTRUCTION AND EXECUTION################

# Network Parameters 
batch_size = 1 # of densecap phrases to use in tensor input per epoch
phraseLength = 5 # of words per phrase. This will become a function of phrase inputs
n_hidden = 128 # hidden layer num of features (# of LSTM cells)
n_layers = 1 #number of stacked layers
# Training Parameters
learning_rate = 0.001
epoch_size = 1
display_step = 1

captions = extract_caption_vectors()
phrases = extract_phrase_vectors()
print phrases
print [[[0] * LEX_DIM] * phraseLength] * batch_size
#phrases = [[[0] * LEX_DIM] * phraseLength] * batch_size

inputs = rn.NetworkInput(phraseLength, LEX_DIM, [phrases, captions], batch_size, epoch_size, training_iters)
params = rn.NetworkParameters(n_hidden, n_layers, learning_rate) 
ann = rn.LSTMNet(inputs, params, [wordDict, invertDict])
ann.run_epoch()

#ann.train()
#phrases = extract_flat_phrase_vectors()

#print len([] * phraseCount)
#print "Phrase %s count: %s" % (phrases, len(phrases))
#print phrases[0]      #image? Why are we nested so deeply
#print len(phrases[0])
#print phrases[0][0]    #phrase - should have length 25
#print len(phrases[0][0])
#print phrases[0][0][0]   #word - length of alphabet
#print len(phrases[0][0][0])
#print phrases[0][0][0][0] #element
#print "Caption %s count: %s" % (type(captions), len(captions))
#Placeholders encode structure and contain training data when we execute the network
#phraseTensor, captionTensor = rn.MakePlaceholderTensors(phraseCount, phraseLength, LEX_DIM)
#print phraseTensor
#print captionTensor
#netWeights, netBiases = rn.MakeLayerParameters(n_hidden, LEX_DIM)
#lstm_net=rn.RNN(phraseTensor, phraseCount, phraseLength, LEX_DIM, n_hidden, netWeights, netBiases)
# Defaults to cross entropy and Adam optimizer (Kingma et al. 2014)
# we will try other cost functions / optimizers later
#cost, optimizer = rn.LossFunction(lstm_net, captionTensor, learning_rate)
#accuracy = rn.AccuracyFunction(lstm_net, captionTensor)
#ongoing = rn.RunModel(lstm_net, phraseTensor, captionTensor, phrases, captions, cost, optimizer, accuracy, training_iters, display_step, wordDict)

