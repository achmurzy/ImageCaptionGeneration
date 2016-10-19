# ImageCaptionGeneration

# We use a pre-trained model and provided lua script to receive data
# Make sure densecap is installed correctly along with torch following
# directions at  https://github.com/myfavouritekk/densecap/blob/master/README.md

# The code depends on the cocopy MS-COCO Python API: https://github.com/pdollar/coco
# Follow the instructions there for installation and make sure to put the installation folder
# on your PYTHONPATH.

# Get MS-COCO data here : http://mscoco.org/dataset/#download
# The repo also assumes the /images is filled with the MS-COCO 2014 training set of images
# /annotations should contain the file "captions_train2014.json" 

# This repository must be next to 'densecap' git clone in a directory. 
# Put an image dataset in 'images' (musicians provided), should be small (less than 5 images) 
# In root of 'densecap' run 'sh scripts/download_pretrained_model.sh'
# Then, run: 
# 'th run_model.lua -input_dir ../ImageCaption/Generation/images -output_vis_dir ../ImageCaptionGeneration/results -output_dir ../ImageCaptionGeneration/results/boxes  -gpu -1'

# Output of densecap is a JSON string containing phrases and boxes and importance scores
# boxes are given as xy coordinate of upper left corner and width/height
# We want to give an id to every phrase, box and score, and store each in a dictionary

# I wrote code that extracts stuff we need from the JSON string, and a method that takes a
# Saliency algorithm to determine which phrases are important. We can start with this.

# To run the code:   python caption_generation.py

# Next steps: 
#   -Input format
#   -Network architecture
#   -Implement in tensorflow and iterate
