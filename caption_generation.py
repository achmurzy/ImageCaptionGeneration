import recurrent_network as rn
import os
import sys
import code
import re
import pickle

#Use if images need re-processing by densecap
#Run as python caption_generation.py 1
processImages = int(sys.argv[1])
read = int(sys.argv[2])
train = int(sys.argv[3])


if read:

    if train:
        fileApp = os.getcwd() + "/train"
    else:
        fileApp = os.getcwd() + "/test"

    phraseFile = open(fileApp + "/phrases", 'r')
    flatPhraseIDs = pickle.load(phraseFile)
    
    capFile = open(fileApp + "/caps", 'r')
    flatCaptionIDs = pickle.load(capFile)

    inFile = open(fileApp + "/inputs", 'r')
    inputs = pickle.load(inFile)

    paramFile = open(fileApp + "/params", 'r')
    params = pickle.load(paramFile)

    encFile = open(fileApp + "/encoder", 'r')
    encoder = pickle.load(encFile)

    decFile = open(fileApp + "/decoder", 'r')
    decoder = pickle.load(decFile)

    import reader
    batchedPhrases, batchedCaptions, epochSize = reader.ptb_id_producer(flatPhraseIDs, 
                flatCaptionIDs, inputs.batch_size, inputs.phraseCount, inputs.phrase_dimension)
    results = rn.NetworkResults(1)
    ann = rn.LSTMNet(inputs, params, results, [encoder, decoder], train)
    ann.run_network(train)

else:

    import densecap_processing as dp

    ##########MS-COCO TRAINING CAPTION EXTRACTION##############
    inputImgCount = 50

    dp.save_full_coco_lexicon(inputImgCount)
    dp.set_coco_dataset(train)

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

    #Build a lexicon and encoder/decoder dictionaries - EDIT :: We load these dictionaries
    #lex = dp.get_coco_lexicon(capDict)                        manually to merge train/test corpora
    #wordDict = {}
    #invertDict = {}
    #dp.build_lookup_lexicon(lex, wordDict, invertDict)

    ###########DENSECAP PHRASE EXTRACTION######################

    if(processImages):
        dp.coco_to_densecap(imgIDs, train)

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
    LEX_DIM = (len(encoder))
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

    flatPhraseIDs = dp.extract_phrase_id_vectors(phraseCount, phraseLength, inputImgCount, phraseCapCorrespondence, image_props, decoder)
    flatCaptionIDs = dp.extract_caption_id_vectors(phraseLength, inputImgCount, decoder, captions)

    inputs = rn.NetworkInput(batch_size, phraseCount, phraseLength, LEX_DIM, batchedPhrases, batchedCaptions, num_epochs, epochSize)
    params = rn.NetworkParameters(n_hidden, n_layers, learning_rate, initializationScale)

    if train:
        fileApp = os.getcwd() + "/train"
    else:
        fileApp = os.getcwd() + "/test"

    phraseFile = open(fileApp + "/phrases", 'w')
    pickle.dump(flatPhraseIDs, phraseFile)
    phraseFile.close()

    capFile = open(fileApp + "/caps", 'w')
    pickle.dump(flatCaptionIDs, capFile)
    capFile.close()

    inputs = open(fileApp + "/inputs", 'w')
    rni = rn.NetworkInput(batch_size, phraseCount, phraseLength, LEX_DIM, None, None, num_epochs, None)
    pickle.dump(rni, inputs)
    inputs.close()

    params = open(fileApp + "/params", 'w')
    rnp = rn.NetworkParameters(n_hidden, n_layers, learning_rate, initializationScale)
    pickle.dump(rnp, params)
    params.close()

    encoder = open(fileApp + "/encoder", 'w')
    renc = wordDict
    pickle.dump(renc, encoder)
    encoder.close()
    
    decoder = open(fileApp + "/decoder", 'w')
    rdec = invertDict
    pickle.dump(rdec, decoder)
    decoder.close()
