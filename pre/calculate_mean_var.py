import numpy as np
import cv2
import os,sys
path='/home/kesci/dataset_fusai/train_val/'
all_B=np.array([])
all_G=np.array([])
all_R=np.array([])
for class_num in os.listdir(path):
    if class_num=='.DS_Store':
        continue
    person_path=path+class_num+'/'
    for img_name in os.listdir(person_path):
        imgpath=person_path+img_name
        img=cv2.imread(imgpath)
        B=img[:,:,0].sum()/96/112
        G=img[:,:,1].sum()/96/112
        R=img[:,:,2].sum()/96/112
        all_B=np.append(all_B,B)
        all_G=np.append(all_G,G)
        all_R=np.append(all_R,R)
mean_B=all_B.mean()
mean_G=all_G.mean()
mean_R=all_R.mean()
var_B=all_B.var()
var_G=all_G.var()
var_R=all_R.var()
print 'mean: RGB', mean_R,mean_G,mean_B
print 'var: RGB', var_R,var_G,var_B
import math
print mean_R/255,mean_G/255,mean_B/255
print math.sqrt(var_R/255/255),math.sqrt(var_G/255/255),math.sqrt(var_B/255/255)