
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
    return images

#takes image info to get various fields
def get_captions(_images, img_index):
    return _images[img_index]['captions']

def get_boxes(_images, img_index):
    return _images[img_index]['boxes']

def get_scores(_images, img_index):
    return _images[img_index]['scores']

