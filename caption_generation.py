import densecap_processing as dp
import salience

#This script gets five images from MS-COCO, prints their training captions
#performs dense captioning, then prints the top five phrases associated with each

#get three img IDs from MS-COCO
imgIDs = dp.get_coco_imgs(5)

#get caption sets for each image
captions = dp.coco_to_captions(imgIDs)
print captions

dp.coco_to_densecap(imgIDs)

results = dp.json_to_dict("results/results.json")
image_props = dp.dict_to_imgs(results)

#for key, value in image_props[0]['scores'].iteritems() :
#    print key, value
#print(results)
#print(image_props[0]['img_name'])
#print(image_props[0])
#print(dp.get_boxes(image_props, 0))
#print(dp.get_scores(image_props, 0))
#print(dp.get_captions(image_props, 0))
print(dp.salient_phrases(image_props[0], lambda: salience.k_top_scores(image_props[0], 5)))
#print(dp.salient_phrases(image_props[0], lambda: salience.false_color_salience(image_props[0], 5)))
