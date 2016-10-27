import densecap_processing as dp

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
