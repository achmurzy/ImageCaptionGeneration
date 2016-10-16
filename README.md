# ImageCaptionGeneration

# We use a pre-trained model and provided lua script to receive data

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

# Next steps: 
#   -Input format
#   -Network architecture
#   -Implement in tensorflow and iterate