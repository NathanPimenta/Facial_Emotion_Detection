import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

datadirectory = "archive/train/"
Classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

for category in Classes:
    path = os.path.join(datadirectory , category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img))
        
        break
    break

img_size = 224  #Image net 224*224 for transfer learning
new_array = cv2.resize(img_array,(img_size,img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()    

training_Data =[] #data array

def create_training_Data():
    for category in Classes:
        path = os.path.join(datadirectory,category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img))
                new_array = cv2.resize(img_array,(img_size,img_size))
                training_Data.append([new_array,class_num]) 
            except Exception as e:
                pass

create_training_Data()

print (len(training_Data))

import random
random.shuffle(training_Data)