import densecap_processing as dp

import nltk
import sys
from nltk.collocations import *
import os

def getTopPhrases(n, phrases):

    try:
        important_phrases = {};

        #create input file with each phrase on a single line
        input_file = open('input', 'w')
        for phrase in phrases:
            input_file.write("%s\n" % phrase)
        input_file.close()
        ##################### Using Bigrams #########################
        bigram_measures = nltk.collocations.BigramAssocMeasures()
        # get current working directory
        cwd = os.getcwd()
        finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words(cwd + '/input'))
        # only bigrams that appear 3+ times
        finder.apply_freq_filter(3)
        # return the 'n' 3-grams with the highest PMI
        top_bigrams = finder.nbest(bigram_measures.pmi, n+30)

        ##################### Using Trigrams #########################
        # trigram_measures = nltk.collocations.TrigramAssocMeasures()
        # # get current working directory
        # cwd = os.getcwd()
        # finder = TrigramCollocationFinder.from_words(nltk.corpus.genesis.words(cwd + '/input'))
        # finder.apply_freq_filter(3)
        # top_bigrams = finder.nbest(trigram_measures.pmi, n + 30)

    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    for item in top_bigrams:
        try:
            search_string = ' '.join(item);
            # combine bigram/trigram with score from densecap
            for phrase in phrases:
                if search_string in phrase and phrase not in important_phrases:
                    important_phrases[phrase] =  True
                    break;
            if len(important_phrases) == n:
                return important_phrases.keys()
        except:
            print 'No problem, keep trying'
    # if keys are less than 'n', then make sure that it has 'n' keys
    keys = important_phrases.keys()
    for i in range(0, n - len(keys)):
        keys.append("");
    return keys

#Takes a saliency function to return training phrases for a given image
#Our network architecture may not require a fixed number of target phrases
def salient_phrases(images, index, saliency):
    phrases = [] 
    for dex in saliency():
        phrases.append(dp.get_captions(images, index)[dex])
    return phrases

## Saliency Methods ## 
## Take image and phrase number k ##
## Returns set of indices to image annotations ##

#The dense captioning software ranks phrases according to their own confidence metric
#We can try using this metric to choose our phrase set
#The function returns the indices of the k-largest confidence scores. These are already sorted
#in the JSON string, so just return the first k elements
def k_top_scores(image, k):
    return range(0, k) 
    
#Returns the k most salient bounding box indices according to the algorithm in Kelleher et al. 2004
#Requires image dimensions - Assumes large, central objects are most salient
#This method becomes more unreliable as lower-ranked phrases are reached -- these are typically
#very large boxes with highly inaccurate descriptions
#Might be useful to use on the phrases ranked highest (top 10?) by the densecap algorithm
def false_color_salience(image, k):
    imgWidth = image['dim'][0]
    imgHeight = image['dim'][1]
    imgCenter = [imgWidth/2, imgHeight/2]
    cornerDist = math.sqrt(math.pow(imgWidth - imgCenter[0], 2) + math.pow(imgHeight - imgCenter[1], 2))
    salience = {}
    count = 0
    #print(image['img_name'])
    #print("Width: ", imgWidth)
    #print("Height: ", imgHeight)
    #print("Corner Distance: ", cornerDist)
    #print("Center: ", imgCenter)
    maxWeight = 0
    for box in image['boxes']:
        #print(count)
        #print(box)
        #print(image['captions'][image['boxes'].index(box)])
        x = box[0]
        y = box[1]
        width = box[2]
        height = box[3]
        weight = 0
        for _x in range(int(math.floor(x)), int(math.floor(x + width))):
            for _y in range(int(math.floor(y)), int(math.floor(y + height))):
                pDis = math.sqrt(math.pow(imgCenter[0] - _x, 2) + math.pow(imgCenter[1] - _y, 2))
                weight += 1 - (pDis / (cornerDist + 1))
        salience[count] = weight
        #print(weight)
        count = count + 1
    phraseIndices = sorted(salience, key=salience.get)
    #print(sorted(salience, key=salience.get))
    #print(phraseIndices)
    #print(phraseIndices[-k:])
    phraseIndices.reverse()
    #print(phraseIndices[:k])
    return phraseIndices[:k]                    
