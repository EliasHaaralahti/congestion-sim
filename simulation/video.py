import cv2
import numpy as np
import glob
import re
 
# helper function to perform sort
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]

img_array = []
filenames = glob.glob('figures/*.png')
filenames.sort(key=num_sort)

for filename in filenames:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

print("processed filenames")
#out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
out = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

for image in img_array:
    out.write(image)
out.release()
print("done")
