import densecap_processing as dp

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
#print(dp.salient_phrases(image_props[0], lambda: dp.k_top_scores(image_props[0], 5)))
print(dp.salient_phrases(image_props[0], lambda: dp.false_color_salience(image_props[0], 5)))
