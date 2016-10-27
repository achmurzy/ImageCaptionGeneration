
# Output of densecap is a JSON string containing phrases and boxes and importance scores
# boxes are given as xy coordinate of upper left corner and width/height
# We want to give an id to every phrase, box and score, and store each in a dictionary

from pycocotools.coco import COCO
import json as js
import subprocess
import math
import os

annFile = 'annotations/captions_train2014.json'
coco = COCO(annFile)

START = '\''
STOP = '.'

#Get some number of imgIDs from MS-COCO
def get_coco_imgs(number):
    imgIDs = sorted(coco.getImgIds())
    numImages = imgIDs[:number]
    return numImages

#Still assumes densecap is in the folder next to ImageCaptionGeneration
#/densecap
#/ImageCaptionGeneration
#    /annotations <= contains MS-COCO 2014 annotations
#    /images <= contains MS-COCO 2014 training images
#    /densecap_images <= input dir to densecap processing
#    /results
#        /boxes <= holds visually annotated images after densecap

#Automatically executes densecap on a set of image Ids from MS-COCO
#automatically writes results to /results
#need to be in /densecap to execute "run_model.lua" because I can't figure
#out how to install lua modules. 
def coco_to_densecap(imgIDs):
    for x in imgIDs:
        command = 'cp images/COCO_train2014_%s.jpg densecap_images'%(str(x).zfill(12))
        print str(x) + " Executing: "+ command
        subprocess.call(command, shell=True)
    os.chdir("../densecap")
    dense_command = "th run_model.lua -input_dir ../ImageCaptionGeneration/densecap_images -output_vis_dir ../ImageCaptionGeneration/results -output_dir ../ImageCaptionGeneration/results/boxes -gpu -1"
    subprocess.call(dense_command, shell=True)
    os.chdir("../ImageCaptionGeneration")
    
#Retrieves set of captions for a set of image Ids from MS-COCO
def coco_to_captions(imgIDs):
    numAnnIDs = coco.getAnnIds(imgIDs)
    return coco.loadAnns(numAnnIDs)

#Returns a single caption associated with each image from the set of annotations
#images have multiple captions which we may be able to take advantage of later
#We also add <START> and <STOP> symbols - we use escaped characters \' and \" respectively
def get_coco_captions(_captions):
    capDict = {}
    for x in _captions:
        if not x['image_id'] in capDict:
            capDict[x['image_id']] = START + x['caption']
    return capDict

#Returns a sorted list of unique words from the set of annotations
def get_coco_lexicon(_captions):
    words = []
    for x in _captions:
        for word in x['caption'].split():
            words.append(word)
    return sorted(set(words))

#Build 2-way lookup tables for encoding and decoding word representations in our neural net
def build_lookup_lexicon(_lexicon, index2word, word2index):
    count = 0;
    for x in _lexicon:
        index2word[count] = x
        word2index[x] = count
        count = count + 1
    # Add special symbols
    index2word[count] = START
    word2index[START] = count
    count = count + 1
    index2word[count] = STOP
    word2index[STOP] = count

#Used to retrieve densecap processing results
#Takes a path to _json filename. Should be "results/results.json" in the local directory.
def json_to_dict(_json):
    json = open(_json, 'r')
    result = js.load(json)
    return result

#parse densecap results for image info
def dict_to_imgs(_result):
    images = {}    
    count = 0
    for img in _result['results']:
        images[count] = img
        count = count + 1
    print "Counted %s images" % count
    return images

#takes image info to get various fields
def get_captions(_images, img_index):
    return _images[img_index]['captions']

def get_boxes(_images, img_index):
    return _images[img_index]['boxes']

def get_scores(_images, img_index):
    return _images[img_index]['scores']

